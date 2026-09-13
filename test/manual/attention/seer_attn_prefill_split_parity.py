"""Parity: split prefill summary path (pool + GEMM + norm/rope) vs a pure-pytorch
reference.

The split path (pool Triton kernel -> cuBLAS batched GEMM -> norm/rope Triton
kernel, all over one locate-in-kernel grid) must match a straightforward pytorch
implementation of the same math (pool max|min|avg -> linear_k -> RMSNorm -> NeoX
RoPE) for the summary cache, and the [max|min|sum] reduction for the rolling
accumulator, to within fp tolerance.

Usage:
    .pixi/envs/default/bin/python test/srt/seer_attn_prefill_split_parity.py
"""

import torch
import triton

from sglang.srt.layers.attention.blocksparse.seer_attn.cache_kernels import (
    _prefill_pool_kernel,
    _prefill_normrope_kernel,
    build_prefill_summary_schedule,
    update_summary_cache_prefill,
    _K_POOL_DUP,
)


def _summary_ref(k_nope, ext_list, rpi, req_to_summary, summary_cache,
                 proj_weight, norm_weight, inv_freq, block_size, eps):
    """Pure-pytorch ground truth for the per-complete-block gated summary:
    pool [max|min|avg] -> linear_k -> RMSNorm(GATE_D) -> NeoX RoPE at block-start."""
    device = k_nope.device
    Hk = k_nope.shape[1]
    GATE_D = summary_cache.shape[-1]
    HALF = GATE_D // 2
    w = proj_weight.to(torch.float32)  # [Hk, 3D, GATE_D]
    nw = norm_weight.to(torch.float32)  # [GATE_D]
    tok = 0
    for i, L in enumerate(ext_list):
        nfull = L // block_size
        for j in range(nfull):
            s = tok + j * block_size
            blk = k_nope[s:s + block_size].float()  # [BN, Hk, D]
            pooled = torch.cat([blk.amax(0), blk.amin(0), blk.mean(0)], dim=-1)  # [Hk, 3D]
            o = torch.einsum("hc,hcg->hg", pooled, w)  # [Hk, GATE_D]
            rms = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
            o = o * rms * nw
            # NeoX RoPE at block-start position j*block_size
            pos = j * block_size
            ang = pos * inv_freq.float()  # [HALF]
            cos, sin = ang.cos(), ang.sin()
            lo, hi = o[:, :HALF], o[:, HALF:]
            out = torch.cat([lo * cos - hi * sin, hi * cos + lo * sin], dim=-1)
            slot = int(req_to_summary[int(rpi[i]), j])
            summary_cache[slot] = out.to(summary_cache.dtype)
        tok += L


def _split(k_nope, schedule, req_to_summary, summary_cache, rolling_buffer, proj_weight, norm_weight, inv_freq, block_size, eps):
    # The public wrapper now drives the whole split path (single-grid pool +
    # locate, GEMM, norm/rope) — call it directly.
    update_summary_cache_prefill(
        schedule, k_nope, summary_cache, rolling_buffer, req_to_summary,
        proj_weight, norm_weight, inv_freq, block_size, eps,
    )


def _rolling_ref(k_nope, ext_list, rpi, block_size, Hk, D, max_reqs):
    """Pure-pytorch reference for the rolling accumulator seed: each request's
    trailing partial block reduced to [max | min | sum]; remainder==0 -> untouched."""
    acc = torch.zeros(max_reqs, Hk, 3 * D, dtype=torch.float32, device=k_nope.device)
    tok = 0
    for i, L in enumerate(ext_list):
        rem = L % block_size
        start = tok + (L // block_size) * block_size
        if rem > 0:
            blk = k_nope[start:start + rem].float()  # [rem, Hk, D]
            r = int(rpi[i])
            acc[r, :, :D] = blk.amax(dim=0)
            acc[r, :, D:2 * D] = blk.amin(dim=0)
            acc[r, :, 2 * D:] = blk.sum(dim=0)
        tok += L
    return acc


def _chunked(k_full_list, ext_list, chunk_size, rpi, block_size,
             req_to_summary, summary_cache, rolling_buffer,
             proj_weight, norm_weight, inv_freq, eps):
    """Drive a block-aligned CHUNKED prefill: feed each request's tokens in
    successive ``chunk_size`` slices (chunk_size a multiple of block_size), with a
    growing ``extend_prefix_lens``.  Requests that finish drop out of later rounds
    (mimicking the scheduler shrinking the extend batch).  ``summary_cache`` and
    ``rolling_buffer`` are written in place across rounds — the end state must equal
    the one-shot whole-sequence build.

    ``k_full_list[i]`` is request ``i``'s own ``[L_i, Hk, D]`` keys.
    """
    device = summary_cache.device
    bsz = len(ext_list)
    prefix = [0] * bsz
    n_rounds = 0
    while any(prefix[i] < ext_list[i] for i in range(bsz)):
        active = [i for i in range(bsz) if prefix[i] < ext_list[i]]
        chunk_lens = [min(chunk_size, ext_list[i] - prefix[i]) for i in active]
        # prefix at call time is always a multiple of chunk_size (hence of
        # block_size) — a request only ever advances by whole chunks until its
        # final (possibly short) chunk, after which it drops out.
        k_chunk = torch.cat(
            [k_full_list[i][prefix[i]:prefix[i] + cl] for i, cl in zip(active, chunk_lens)]
        )
        ext_round = torch.tensor(chunk_lens, dtype=torch.int64, device=device)
        pref_round = torch.tensor([prefix[i] for i in active], dtype=torch.int64, device=device)
        rpi_round = rpi[active]
        sched = build_prefill_summary_schedule(
            ext_round, rpi_round, block_size,
            extend_seq_lens_cpu=ext_round.cpu(),
            extend_prefix_lens=pref_round,
            extend_prefix_lens_cpu=pref_round.cpu(),
        )
        update_summary_cache_prefill(
            sched, k_chunk, summary_cache, rolling_buffer, req_to_summary,
            proj_weight, norm_weight, inv_freq, block_size, eps,
        )
        for idx, i in enumerate(active):
            prefix[i] += chunk_lens[idx]
        n_rounds += 1
    return n_rounds


def main():
    device = "cuda"
    torch.manual_seed(0)
    block_size = 64
    Hk, D, GATE_D = 8, 128, 128
    eps = 1e-6
    dtype = torch.bfloat16

    cases = [
        [130, 64, 200, 63],     # mixed: partials + full blocks
        [256, 256, 256, 256],   # all equal, all full
        [4096, 100],            # one long one short
        [64],                   # single request, exactly one block
        [40000, 8192, 16384, 500],  # large ragged
        [30, 50, 10],           # ALL partial: total_blocks == 0 (rolling-only path)
    ]

    n_ok = n_fail = 0
    for ext_list in cases:
        bsz = len(ext_list)
        ext = torch.tensor(ext_list, dtype=torch.int64, device=device)
        rpi = torch.arange(bsz, dtype=torch.int64, device=device)
        max_blocks = (max(ext_list) // block_size) + 1
        max_reqs = bsz + 2
        # summary slots: assign distinct rows per (req, block); slot 0 reserved
        req_to_summary = torch.zeros((max_reqs, max_blocks), dtype=torch.int32, device=device)
        nslot = 1
        for i in range(bsz):
            nb = ext_list[i] // block_size
            for j in range(nb):
                req_to_summary[i, j] = nslot
                nslot += 1
        summary_pool = nslot + 1

        num_tokens = int(ext.sum())
        k_nope = torch.randn(num_tokens, Hk, D, device=device, dtype=dtype)
        proj_weight = torch.randn(Hk, _K_POOL_DUP * D, GATE_D, device=device, dtype=dtype) * 0.05
        norm_weight = torch.randn(GATE_D, device=device, dtype=dtype) * 0.1 + 1.0
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, GATE_D // 2, device=device, dtype=torch.float32) * 2 / GATE_D))

        sched = build_prefill_summary_schedule(ext, rpi, block_size,
                                               extend_seq_lens_cpu=ext.cpu())

        sc_ref = torch.zeros(summary_pool, Hk, GATE_D, device=device, dtype=dtype)
        sc_split = torch.zeros(summary_pool, Hk, GATE_D, device=device, dtype=dtype)
        roll_split = torch.zeros(max_reqs, Hk, 3 * D, device=device, dtype=torch.float32)
        _summary_ref(k_nope, ext_list, rpi, req_to_summary, sc_ref, proj_weight, norm_weight, inv_freq, block_size, eps)
        _split(k_nope, sched, req_to_summary, sc_split, roll_split, proj_weight, norm_weight, inv_freq, block_size, eps)

        diff = (sc_ref.float() - sc_split.float()).abs()
        max_abs = diff.max().item()
        # rolling-accumulator seed (merged into the pool kernel) vs pytorch ref
        roll_ref = _rolling_ref(k_nope, ext_list, rpi, block_size, Hk, D, max_reqs)
        roll_max = (roll_split - roll_ref).abs().max().item()
        # bf16 summary store + different projection reduction order -> allow a
        # few bf16 ULPs.  bf16 eps ~ 0.0078; values are O(1) after RMSNorm.
        tol = 0.02
        # rolling acc is fp32 in/out; bf16 k_nope is the only lossy step, so the
        # reduction matches the ref to fp32 round-off.
        roll_tol = 1e-3
        if max_abs <= tol and roll_max <= roll_tol:
            n_ok += 1
            print(f"OK   ext={ext_list} total_blocks={sched.total_blocks} "
                  f"summary_max={max_abs:.5f} rolling_max={roll_max:.6f}")
        else:
            n_fail += 1
            print(f"FAIL ext={ext_list} total_blocks={sched.total_blocks} "
                  f"summary_max={max_abs:.5f} (tol={tol}) rolling_max={roll_max:.6f} (tol={roll_tol})")

    print(f"\n{n_ok} OK, {n_fail} failed")

    # ---- CHUNKED prefill parity (the core correctness gate for chunked support) --
    # Feed a request's tokens in block_size-aligned chunks with a growing
    # extend_prefix_lens; the accumulated summary + rolling caches MUST equal both
    # (a) the one-shot whole-sequence build and (b) the pure-pytorch _summary_ref /
    # _rolling_ref.  chunk_size is a multiple of block_size so no complete block
    # straddles a chunk (the alignment the kernel requires).
    print("\n---- CHUNKED prefill parity ----")
    ck_ok = ck_fail = 0
    # (ext_list, chunk_size): chunk_size % block_size == 0.  Mixes single-request
    # and multi-request (ragged, requests finishing at different rounds) batches.
    chunk_cases = [
        ([8192], 512),            # single req, 16 chunks
        ([524288], 8192),         # long single req (the 512k benchmark shape), 64 chunks
        ([1024, 4096], 512),      # ragged: req0 done in 2 rounds, req1 in 8
        ([320, 640, 128], 128),   # short ragged, chunk == 2 blocks
        ([200, 4096, 63], 256),   # partials at the end of some reqs
        ([16384, 16384], 4096),   # two equal long reqs
    ]
    for ext_list, chunk_size in chunk_cases:
        assert chunk_size % block_size == 0
        bsz = len(ext_list)
        rpi = torch.arange(bsz, dtype=torch.int64, device=device)
        max_blocks = (max(ext_list) // block_size) + 1
        max_reqs = bsz + 2
        req_to_summary = torch.zeros((max_reqs, max_blocks), dtype=torch.int32, device=device)
        nslot = 1
        for i in range(bsz):
            nb = ext_list[i] // block_size
            for j in range(nb):
                req_to_summary[i, j] = nslot
                nslot += 1
        summary_pool = nslot + 1

        # Per-request keys; concatenation is the one-shot / ref input.
        k_full_list = [torch.randn(L, Hk, D, device=device, dtype=dtype) for L in ext_list]
        k_cat = torch.cat(k_full_list)
        proj_weight = torch.randn(Hk, _K_POOL_DUP * D, GATE_D, device=device, dtype=dtype) * 0.05
        norm_weight = torch.randn(GATE_D, device=device, dtype=dtype) * 0.1 + 1.0
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, GATE_D // 2, device=device, dtype=torch.float32) * 2 / GATE_D))

        # (a) one-shot whole-sequence build
        sched_1 = build_prefill_summary_schedule(
            torch.tensor(ext_list, dtype=torch.int64, device=device), rpi, block_size,
            extend_seq_lens_cpu=torch.tensor(ext_list),
        )
        sc_oneshot = torch.zeros(summary_pool, Hk, GATE_D, device=device, dtype=dtype)
        roll_oneshot = torch.zeros(max_reqs, Hk, 3 * D, device=device, dtype=torch.float32)
        update_summary_cache_prefill(
            sched_1, k_cat, sc_oneshot, roll_oneshot, req_to_summary,
            proj_weight, norm_weight, inv_freq, block_size, eps,
        )

        # (b) chunked build (this is what we're validating)
        sc_chunk = torch.zeros(summary_pool, Hk, GATE_D, device=device, dtype=dtype)
        roll_chunk = torch.zeros(max_reqs, Hk, 3 * D, device=device, dtype=torch.float32)
        rounds = _chunked(
            k_full_list, ext_list, chunk_size, rpi, block_size,
            req_to_summary, sc_chunk, roll_chunk,
            proj_weight, norm_weight, inv_freq, eps,
        )

        # (c) pure-pytorch reference
        sc_ref = torch.zeros(summary_pool, Hk, GATE_D, device=device, dtype=dtype)
        _summary_ref(k_cat, ext_list, rpi, req_to_summary, sc_ref,
                     proj_weight, norm_weight, inv_freq, block_size, eps)
        roll_ref = _rolling_ref(k_cat, ext_list, rpi, block_size, Hk, D, max_reqs)

        d_vs_oneshot = (sc_chunk.float() - sc_oneshot.float()).abs().max().item()
        d_vs_ref = (sc_chunk.float() - sc_ref.float()).abs().max().item()
        r_vs_oneshot = (roll_chunk - roll_oneshot).abs().max().item()
        r_vs_ref = (roll_chunk - roll_ref).abs().max().item()
        tol, roll_tol = 0.02, 1e-3
        ok = (d_vs_oneshot <= tol and d_vs_ref <= tol
              and r_vs_oneshot <= roll_tol and r_vs_ref <= roll_tol)
        if ok:
            ck_ok += 1
            print(f"OK   ext={ext_list} chunk={chunk_size} rounds={rounds} "
                  f"sum(vs1shot={d_vs_oneshot:.5f} vsref={d_vs_ref:.5f}) "
                  f"roll(vs1shot={r_vs_oneshot:.6f} vsref={r_vs_ref:.6f})")
        else:
            ck_fail += 1
            print(f"FAIL ext={ext_list} chunk={chunk_size} rounds={rounds} "
                  f"sum(vs1shot={d_vs_oneshot:.5f} vsref={d_vs_ref:.5f} tol={tol}) "
                  f"roll(vs1shot={r_vs_oneshot:.6f} vsref={r_vs_ref:.6f} tol={roll_tol})")
    print(f"\nchunked: {ck_ok} OK, {ck_fail} failed")
    n_ok += ck_ok
    n_fail += ck_fail

    # ---- micro-bench: per-stage split timing + fair fused comparison ----
    # Reports the three split stages (pool+locate / GEMM / norm+rope) separately,
    # plus the old fused single-kernel timed fairly (prebuilt block arrays, so
    # only the kernel is timed, not the host array build).
    import triton as _triton

    def _bench(fn, iters=300):
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1000.0

    print("\n---- prefill summary build timing (us) ----")
    print(f"{'ext':28} {'blocks':>6} {'pool':>7} {'gemm':>7} {'nrope':>7} {'split':>7}")
    for ext_list in ([8192] * 4, [32768], [16384] * 2, [40000, 8192, 16384, 500]):
        bsz = len(ext_list)
        ext = torch.tensor(ext_list, dtype=torch.int64, device=device)
        rpi = torch.arange(bsz, dtype=torch.int64, device=device)
        max_blocks = (max(ext_list) // block_size) + 1
        req_to_summary = torch.zeros((bsz + 2, max_blocks), dtype=torch.int32, device=device)
        nslot = 1
        for i in range(bsz):
            for j in range((ext_list[i] + block_size - 1) // block_size):
                req_to_summary[i, j] = nslot; nslot += 1
        num_tokens = int(ext.sum())
        k_nope = torch.randn(num_tokens, Hk, D, device=device, dtype=dtype)
        proj_weight = torch.randn(Hk, _K_POOL_DUP * D, GATE_D, device=device, dtype=dtype) * 0.05
        norm_weight = torch.randn(GATE_D, device=device, dtype=dtype) * 0.1 + 1.0
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, GATE_D // 2, device=device, dtype=torch.float32) * 2 / GATE_D))
        sched = build_prefill_summary_schedule(ext, rpi, block_size, extend_seq_lens_cpu=ext.cpu())
        sc = torch.zeros(nslot + 1, Hk, GATE_D, device=device, dtype=dtype)
        roll = torch.zeros(bsz + 2, Hk, 3 * D, device=device, dtype=torch.float32)
        tb = sched.total_blocks
        BSZ_POW2 = _triton.next_power_of_2(bsz)
        BLOCK_D = _triton.next_power_of_2(D)
        HALF = GATE_D // 2; HALF_POW2 = _triton.next_power_of_2(HALF)
        pooled = torch.zeros((tb, Hk, _K_POOL_DUP * D), dtype=torch.float32, device=device)
        proj = torch.empty((tb, Hk, GATE_D), dtype=torch.float32, device=device)

        def stage_pool():
            _prefill_pool_kernel[(tb, Hk)](
                k_nope, k_nope.stride(0), k_nope.stride(1),
                sched.cum_block_cnt, sched.cu_seqlens, sched.rpi32,
                pooled, pooled.stride(0), pooled.stride(1),
                roll, roll.stride(0), roll.stride(1),
                BSZ=bsz, BSZ_POW2=BSZ_POW2, BLOCK_N=block_size, D=D, BLOCK_D=BLOCK_D,
            )

        def stage_gemm():
            torch.einsum("nhc,hcg->nhg", pooled, proj_weight.to(torch.float32))

        def stage_nrope():
            _prefill_normrope_kernel[(tb, Hk)](
                proj, proj.stride(0), proj.stride(1),
                sched.cum_block_cnt, sched.cu_seqlens, sched.rpi32,
                sched.prefix_blocks,
                req_to_summary, req_to_summary.stride(0), req_to_summary.stride(1),
                norm_weight, inv_freq,
                sc, sc.stride(0), sc.stride(1),
                eps, BSZ=bsz, BSZ_POW2=BSZ_POW2, BLOCK_N=block_size,
                GATE_D=GATE_D, HALF=HALF, HALF_POW2=HALF_POW2,
            )

        run_split = lambda: _split(k_nope, sched, req_to_summary, sc, roll, proj_weight, norm_weight, inv_freq, block_size, eps)

        t_pool = _bench(stage_pool)
        t_gemm = _bench(stage_gemm)
        t_nrope = _bench(stage_nrope)
        t_split = _bench(run_split)
        print(f"{str(ext_list)[:28]:28} {tb:6d} {t_pool:7.1f} {t_gemm:7.1f} "
              f"{t_nrope:7.1f} {t_split:7.1f}")

    return n_fail


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
