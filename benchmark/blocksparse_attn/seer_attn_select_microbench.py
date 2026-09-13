"""Micro-benchmark: threshold extraction in _select_from_scores_kernel.

Isolated A/B of the ONLY line changed in the gather refactor — extracting the
n_pick-th largest score (the selection threshold) from the sorted scores:

  OLD: thr = tl.sum(tl.where(blk == thr_idx, sorted_desc, 0.0))   # ~log2(NBLK)
       (a warp-shuffle reduction tree to pull out one lane, then broadcast)
  NEW: thr = tl.gather(sorted_desc, [n_pick-1]*NBLK, axis=0)       # 1 shuffle
       (single-lane warp-shuffle broadcast; already the [NBLK] broadcast value)

Both variants are otherwise byte-identical to the production kernel (same clear,
same tl.sort, same gt/eq/cumsum selection + scatter), so the delta isolates the
threshold-extraction cost.  Run on real decode shapes: H=8 kv-heads, NBLK in
{128, 512, 1024} (≈ 8K / 32K / 40K ctx at block_size=64), bsz in {1, 8, 32}.

Usage:
    .pixi/envs/default/bin/python test/srt/seer_attn_select_microbench.py
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Two kernels, identical except for the threshold-extraction stanza.
# --------------------------------------------------------------------------
def _make_select_kernel(use_gather: bool):
    @triton.jit
    def _kernel(
        scores_ptr,
        scores_stride_b,
        scores_stride_h,
        seq_lens_ptr,
        active_ptr,
        active_stride_b,
        active_stride_h,
        block_size: tl.constexpr,
        topk: tl.constexpr,
        n_pick: tl.constexpr,
        num_init_blocks: tl.constexpr,
        num_local_blocks: tl.constexpr,
        NBLK: tl.constexpr,
        USE_GATHER: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int32)
        current_block = (seq_len - 1) // block_size
        num_prev = current_block
        local_start = num_prev - num_local_blocks

        blk = tl.arange(0, NBLK)
        tl.store(
            active_ptr + pid_b * active_stride_b + pid_h * active_stride_h + blk,
            tl.full([NBLK], -1, dtype=tl.int32),
            mask=blk < topk,
        )
        if num_prev <= 0:
            return

        acc = tl.load(
            scores_ptr + pid_b * scores_stride_b + pid_h * scores_stride_h + blk,
        )

        # ---- threshold extraction (the A/B stanza) ----
        if USE_GATHER:
            if n_pick > 0:
                sorted_desc = tl.sort(acc, descending=True)
                thr = tl.gather(
                    sorted_desc, tl.full([NBLK], n_pick - 1, tl.int32), axis=0
                )
            else:
                thr = tl.full([NBLK], float("inf"), tl.float32)
        else:
            sorted_desc = tl.sort(acc, descending=True)
            thr_idx = n_pick - 1 if n_pick > 0 else 0
            thr_s = tl.sum(tl.where(blk == thr_idx, sorted_desc, 0.0))
            if n_pick <= 0:
                thr_s = float("inf")
            thr = thr_s + 0.0 * blk  # broadcast scalar -> [NBLK] (matches new path)
        # ------------------------------------------------

        neg_inf = float("-inf")
        gt = acc > thr
        eq = (acc == thr) & (thr > neg_inf)
        n_gt = tl.sum(gt.to(tl.int32))
        need = n_pick - n_gt
        eq_rank = tl.cumsum(eq.to(tl.int32)) - eq.to(tl.int32)
        picked = gt | (eq & (eq_rank < need))

        sel = (
            ((blk < num_init_blocks) & (blk < num_prev))
            | ((blk >= local_start) & (blk < num_prev))
            | picked
        )
        sel_i = sel.to(tl.int32)
        pos = tl.cumsum(sel_i) - sel_i
        tl.store(
            active_ptr + pid_b * active_stride_b + pid_h * active_stride_h + pos,
            blk.to(tl.int32),
            mask=sel & (pos < topk),
        )

    return _kernel


_KERNEL = _make_select_kernel(True)  # same jit fn; USE_GATHER picks the branch


def _bench_one(bsz, H, NBLK, topk, block_size, num_init, num_local, use_gather, iters=2000):
    device = "cuda"
    n_pick = topk - (num_init + num_local)
    # Real-ish scores: candidates finite, rest -inf.  All blocks valid (num_prev
    # large) so the selection does full work.
    torch.manual_seed(0)
    scores = torch.randn(bsz, H, NBLK, dtype=torch.float32, device=device)
    # seq_len chosen so num_prev = NBLK-1 (every block id a candidate, no padding)
    seq_lens = torch.full((bsz,), (NBLK - 1) * block_size + 1, dtype=torch.int32, device=device)
    active = torch.empty(bsz, H, topk, dtype=torch.int32, device=device)

    def run():
        _KERNEL[(bsz, H)](
            scores, scores.stride(0), scores.stride(1),
            seq_lens,
            active, active.stride(0), active.stride(1),
            block_size=block_size, topk=topk, n_pick=n_pick,
            num_init_blocks=num_init, num_local_blocks=num_local,
            NBLK=NBLK, USE_GATHER=use_gather,
        )

    # warmup / autotune
    for _ in range(50):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    ms_total = start.elapsed_time(end)
    return ms_total / iters * 1000.0  # us/call


def main():
    H = 8
    block_size = 64
    # (NBLK, topk) pairs ~ 8K / 32K / 40K ctx with budget=2048 (topk=32), plus
    # extreme NBLK to probe the single-program tl.sort/tl.cumsum scaling wall
    # (NBLK=8192-16384 ~ 512K-1M token context at block_size=64).
    shapes = [(128, 32), (512, 32), (1024, 32),
              (2048, 32), (4096, 32), (8192, 32), (16384, 32)]
    batches = [1, 8, 32]
    num_init, num_local = 1, 0

    print(f"H={H} block_size={block_size} num_init={num_init} num_local={num_local}")
    print(f"{'NBLK':>6} {'topk':>5} {'bsz':>4} {'where+sum us':>13} {'gather us':>11} {'speedup':>8}")
    print("-" * 56)
    for NBLK, topk in shapes:
        for bsz in batches:
            try:
                t_old = _bench_one(bsz, H, NBLK, topk, block_size, num_init, num_local, False)
                t_new = _bench_one(bsz, H, NBLK, topk, block_size, num_init, num_local, True)
                sp = t_old / t_new if t_new > 0 else float("nan")
                print(f"{NBLK:6d} {topk:5d} {bsz:4d} {t_old:13.3f} {t_new:11.3f} {sp:7.3f}x")
            except Exception as e:
                msg = str(e).strip().splitlines()[-1][:60] if str(e).strip() else type(e).__name__
                print(f"{NBLK:6d} {topk:5d} {bsz:4d}   FAILED: {msg}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
