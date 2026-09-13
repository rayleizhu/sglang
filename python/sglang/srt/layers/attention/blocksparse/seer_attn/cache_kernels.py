# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Triton kernels for SeerAttention-R *prefill* (extend) cache maintenance.

During extend the attention is dense, but the AttnGate still needs its two
auxiliary caches seeded for the subsequent decode steps:

* **summary cache** — one *gated compressed-K* vector per **complete** logical
  KV block (width ``gate_hidden_size``).  Built by pooling the block's raw
  (pre-RoPE, post-k_norm) keys (max|min|avg), projecting with the gate K-branch
  weight, RMSNorm, and applying NeoX RoPE at the *block-start* position.
* **rolling accumulator** — a per-request running ``[max | min | sum]`` reduction
  (fp32, width ``3 * head_dim``) of the raw keys of the trailing **partial**
  block of each request.  Seeded here from the partial block's keys; folded
  forward one key per decode step.  ``avg`` is recovered as ``sum / block_size``
  at fill time.

This module replaces the per-request Python loop in
``SeerAttnBackend._build_summaries_extend`` with one host entry point,
:func:`update_summary_cache_prefill`, that computes a minimal per-request
schedule once (just the CEIL block-count and token cumsums) and runs a three-stage
split path over a single ``total_blocks`` grid:

* :func:`_prefill_pool_kernel` — pool one block.  Each program LOCATES itself
  (:func:`_locate_block`) — recovers ``(batch_id, intra-req block id, is_partial)``
  from ``cum_block_cnt`` + ``cu_seqlens`` — so the host passes only per-request
  boundaries, not flattened per-block arrays.  Complete blocks pool ``[max|min|avg]``
  into ``pooled`` (for the GEMM); each request's trailing partial block pools
  ``[max|min|sum]`` straight into the rolling accumulator.
* a batched cuBLAS GEMM (``pooled @ attngate_linear_k``) — the bandwidth-heavy
  linear_k, weight loaded once and reused across all blocks.
* :func:`_prefill_normrope_kernel` — RMSNorm + NeoX RoPE at the block-start
  position, written to each block's summary slot (looked up in-kernel; partial
  blocks early-return).

The compress+RoPE math (``_compress_and_rope``) is factored as a ``triton.jit``
device function reused by the *decode*-time summary-update kernel
(:func:`_decode_summary_kernel`, graph-safe, grid ``(bsz, kv_heads)``, masked
store gated on ``seq_len % block_size == 0``) for identical numerics; its
norm+rope half is further split out as :func:`_norm_and_rope` for the prefill
norm/rope kernel (whose projection is the separate GEMM).

CUDA-graph note: SGLang only captures ``ForwardMode.DECODE``; extend is never
captured, so these kernels have no graph constraints.  The host-side schedule
build below is therefore free to use ``.item()``/cumsum.

RoPE convention: NeoX full-width (``cat(freqs, freqs)`` + rotate_half),
computed in fp32 from ``rope_inv_freqs`` to match
:func:`sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate.apply_rotary_pos_emb_seer`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# K-branch pooling is fixed to max|min|avg (Kmaxminavg), concatenated in this
# order, matching the trained ``attngate_linear_k`` weight layout.
_K_POOL_DUP = 3


# ---------------------------------------------------------------------------
# Shared device helper: pooled-feature -> compressed-K (proj + RMSNorm + RoPE)
# ---------------------------------------------------------------------------
@triton.jit
def _compress_and_rope(
    kmax,  # [BLOCK_D] fp32  pooled (max) over the block, padded d-lanes = 0
    kmin,  # [BLOCK_D] fp32
    kavg,  # [BLOCK_D] fp32
    w_ptr,  # proj weight [Hk, 3*D, GATE_D] base pointer
    w_off_head,  # head offset into w_ptr (h * stride_head), int
    sw_c,  # weight stride along the 3*D (in-channel) axis
    sw_g,  # weight stride along the GATE_D (out) axis
    norm_ptr,  # [GATE_D] RMSNorm weight
    inv_freq_ptr,  # [HALF] rope inverse frequencies (fp32)
    pos,  # scalar int  RoPE position (block-start token position)
    eps,  # RMSNorm epsilon (fp32 scalar)
    D: tl.constexpr,  # real head_dim (pooled in-channel width per pool)
    GATE_D: tl.constexpr,  # gate hidden size
    HALF: tl.constexpr,  # GATE_D // 2
    BLOCK_D: tl.constexpr,  # next_pow2(D)
    HALF_POW2: tl.constexpr,  # next_pow2(HALF)
):
    """pooled (max|min|avg) -> linear_k -> knorm -> NeoX RoPE.

    Returns ``(out_lo, out_hi)`` each ``[HALF_POW2]`` fp32 — the rotated first
    and second halves of the ``GATE_D``-wide compressed-K vector.  Padded lanes
    (``offs_half >= HALF``) are unspecified; the caller masks them on store.
    """
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, HALF_POW2)
    d_mask = offs_d < D
    h_mask = offs_h < HALF

    # --- linear_k: o1 = pooled @ W[:, :HALF], o2 = pooled @ W[:, HALF:] ---
    # pooled is the concat [kmax | kmin | kavg] of width 3*D, so we accumulate
    # the three D-wide contributions against their weight sub-blocks (the
    # max/min/avg sub-blocks live at in-channel rows [0,D), [D,2D), [2D,3D)).
    o1 = tl.zeros([HALF_POW2], dtype=tl.float32)
    o2 = tl.zeros([HALF_POW2], dtype=tl.float32)
    for pool_i in tl.static_range(3):
        kv = tl.where(pool_i == 0, kmax, tl.where(pool_i == 1, kmin, kavg))
        c = pool_i * D + offs_d  # in-channel rows for this pool
        w_base = w_ptr + w_off_head + c[:, None] * sw_c
        w_lo = tl.load(
            w_base + offs_h[None, :] * sw_g,
            mask=d_mask[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w_hi = tl.load(
            w_base + (offs_h[None, :] + HALF) * sw_g,
            mask=d_mask[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        o1 += tl.sum(kv[:, None] * w_lo, axis=0)
        o2 += tl.sum(kv[:, None] * w_hi, axis=0)

    # --- RMSNorm over the full GATE_D width (fp32 variance, then weight) ---
    sumsq = tl.sum(tl.where(h_mask, o1 * o1, 0.0)) + tl.sum(
        tl.where(h_mask, o2 * o2, 0.0)
    )
    inv_rms = 1.0 / tl.sqrt(sumsq / GATE_D + eps)
    w_lo_n = tl.load(norm_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    w_hi_n = tl.load(norm_ptr + offs_h + HALF, mask=h_mask, other=0.0).to(tl.float32)
    o1 = o1 * inv_rms * w_lo_n
    o2 = o2 * inv_rms * w_hi_n

    # --- NeoX RoPE (full-width cos/sin == cat(cos_h, cos_h); rotate_half) ---
    inv_freq = tl.load(inv_freq_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    angle = pos.to(tl.float32) * inv_freq
    cos_h = tl.cos(angle)
    sin_h = tl.sin(angle)
    out_lo = o1 * cos_h - o2 * sin_h
    out_hi = o2 * cos_h + o1 * sin_h
    return out_lo, out_hi


@triton.jit
def _norm_and_rope(
    o1,  # [HALF_POW2] fp32  projected first half (pre-norm), padded lanes = 0
    o2,  # [HALF_POW2] fp32  projected second half
    norm_ptr,  # [GATE_D] RMSNorm weight
    inv_freq_ptr,  # [HALF] rope inverse frequencies (fp32)
    pos,  # scalar int  RoPE position
    eps,  # RMSNorm epsilon (fp32 scalar)
    GATE_D: tl.constexpr,
    HALF: tl.constexpr,
    HALF_POW2: tl.constexpr,
):
    """RMSNorm (over full GATE_D) + NeoX RoPE — the second half of
    :func:`_compress_and_rope`, factored out so the split prefill path (where the
    linear projection is a separate cuBLAS GEMM) can reuse the identical
    norm+rope numerics on the already-projected ``(o1, o2)``.

    Returns ``(out_lo, out_hi)`` each ``[HALF_POW2]`` fp32; padded lanes
    (``offs_half >= HALF``) are unspecified (caller masks on store).
    """
    offs_h = tl.arange(0, HALF_POW2)
    h_mask = offs_h < HALF

    sumsq = tl.sum(tl.where(h_mask, o1 * o1, 0.0)) + tl.sum(
        tl.where(h_mask, o2 * o2, 0.0)
    )
    inv_rms = 1.0 / tl.sqrt(sumsq / GATE_D + eps)
    w_lo_n = tl.load(norm_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    w_hi_n = tl.load(norm_ptr + offs_h + HALF, mask=h_mask, other=0.0).to(tl.float32)
    o1 = o1 * inv_rms * w_lo_n
    o2 = o2 * inv_rms * w_hi_n

    inv_freq = tl.load(inv_freq_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    angle = pos.to(tl.float32) * inv_freq
    cos_h = tl.cos(angle)
    sin_h = tl.sin(angle)
    out_lo = o1 * cos_h - o2 * sin_h
    out_hi = o2 * cos_h + o1 * sin_h
    return out_lo, out_hi


# ---------------------------------------------------------------------------
# Split prefill: pool (Triton) -> linear_k (cuBLAS batched GEMM) -> norm/RoPE
# (Triton).  The GEMM is the bandwidth-heavy step (every block reuses the same
# [3*D, GATE_D] gate weight); doing it as one matmul lets cuBLAS load the weight
# once and reuse it across all blocks, vs each (block, head) program re-reading
# it from HBM as the old fused _prefill_summary_kernel did.
#
# Single grid axis `total_blocks` (CEIL: each request's trailing partial block is
# its last block).  Each program LOCATES itself — recovers (batch_id, intra-req
# block id, is_partial) from `cum_block_cnt` + `cu_seqlens` via _locate_block —
# so the host passes only per-request boundaries ([bsz]-sized), not flattened
# per-block [total_blocks] arrays.  Complete blocks pool [max|min|avg] -> `pooled`
# (-> GEMM -> summary); the trailing partial block of each request pools
# [max|min|sum] -> the rolling accumulator (subsuming the old _seed_rolling_kernel).
# ---------------------------------------------------------------------------
@triton.jit
def _locate_block(
    blk,  # global block id (program_id 0)
    cum_block_cnt_ptr,  # [bsz+1] int32 exclusive cumsum of per-req CEIL block count
    cu_seqlens_ptr,  # [bsz+1] int32 exclusive cumsum of extend_seq_lens (token starts)
    BSZ: tl.constexpr,  # bsz
    BSZ_POW2: tl.constexpr,  # next_pow2(bsz)
    BLOCK_N: tl.constexpr,  # block_size
):
    """Recover a global block's owner request and intra-request position.

    Returns ``(batch_id, blk_tok_start, intra_req_blk_id, remainder, is_partial)`` where:
      * ``batch_id``    — request that owns ``blk`` (the i with
                          ``cum_block_cnt[i] <= blk < cum_block_cnt[i+1]``);
      * ``blk_tok_start``   — first token row of this block in the flat k buffer
                          (``cu_seqlens[batch_id] + intra_req_blk_id*BLOCK_N``);
      * ``intra_req_blk_id``     — block index within the request (``blk - cum_block_cnt[batch_id]``);
      * ``remainder``   — valid tokens in this block (``BLOCK_N`` if complete);
      * ``is_partial``  — blk_tok_end token exceeds the request's token-end, i.e.
                          this is the trailing partial block.
    The search is a single masked reduction over the (small) ``[BSZ_POW2]`` lane
    tile — ``batch_id = #{i : cum_block_cnt[i+1] <= blk}``.
    """
    idx = tl.arange(0, BSZ_POW2)
    valid = idx < BSZ
    # cum_block_cnt[1:] are the per-request UPPER bounds; count how many requests
    # end at or before `blk` -> that many requests come strictly before us.
    upper = tl.load(cum_block_cnt_ptr + 1 + idx, mask=valid, other=0)
    batch_id = tl.sum(((upper <= blk) & valid).to(tl.int32))

    base_blk = tl.load(cum_block_cnt_ptr + batch_id)
    intra_req_blk_id = blk - base_blk # intra-request block index
    req_tok_start = tl.load(cu_seqlens_ptr + batch_id) # token start of the request in the ragged buffer
    req_tok_end = tl.load(cu_seqlens_ptr + batch_id + 1) # token end of the request in the ragged buffer
    blk_tok_start = req_tok_start + intra_req_blk_id * BLOCK_N # token start of the block in the ragged buffer
    # `remainder` is the valid token count of this block; equivalently
    # min(req_tok_end, blk_tok_start + BLOCK_N) - blk_tok_start.  We keep it as an
    # explicit count because the kernels need it directly as the n-axis mask bound
    # (offs_n < remainder); the min() form would just be remainder under a
    # different name.
    blk_tok_end = blk_tok_start + BLOCK_N  # ASSUMED end if the block were complete
    is_partial = blk_tok_end > req_tok_end
    remainder = tl.where(is_partial, req_tok_end - blk_tok_start, BLOCK_N)
    return batch_id, blk_tok_start, intra_req_blk_id, remainder, is_partial


@triton.jit
def _prefill_pool_kernel(
    k_ptr,  # [num_tokens, Hk, D] raw pre-RoPE post-k_norm keys
    k_stride_tok,
    k_stride_head,
    cum_block_cnt_ptr,  # [bsz+1] int32
    cu_seqlens_ptr,  # [bsz+1] int32
    req_pool_indices_ptr,  # [bsz] int32  (for the rolling-acc destination)
    pooled_ptr,  # [total_blocks, Hk, 3*D] fp32  out: [max | min | avg] (complete blocks)
    p_stride_blk,
    p_stride_head,
    acc_ptr,  # [max_num_reqs, Hk, 3*D] fp32  rolling [max | min | sum] (partial blocks)
    acc_stride_req,
    acc_stride_head,
    BSZ: tl.constexpr,
    BSZ_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,  # == block_size
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Pool one block.  Complete block -> [max|min|avg] in ``pooled`` (feeds the
    GEMM); trailing partial block -> [max|min|sum] in the rolling accumulator.

    Grid ``(total_blocks, Hk)``; each program locates itself via ``_locate_block``.
    """
    blk = tl.program_id(0)
    head = tl.program_id(1)

    batch_id, blk_tok_start, intra_req_blk_id, remainder, is_partial = _locate_block(
        blk, cum_block_cnt_ptr, cu_seqlens_ptr, BSZ, BSZ_POW2, BLOCK_N
    )

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    n_mask = offs_n < remainder  # all True for a complete block (remainder==BLOCK_N)

    k = tl.load(
        k_ptr
        + (blk_tok_start.to(tl.int64) + offs_n)[:, None] * k_stride_tok
        + head * k_stride_head
        + offs_d[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if is_partial:
        # trailing partial block -> rolling accumulator [max | min | sum].
        kmax = tl.max(tl.where(n_mask[:, None], k, float("-inf")), axis=0)
        kmin = tl.min(tl.where(n_mask[:, None], k, float("inf")), axis=0)
        ksum = tl.sum(tl.where(n_mask[:, None], k, 0.0), axis=0)
        req_pool_idx = tl.load(req_pool_indices_ptr + batch_id).to(tl.int64)
        base = acc_ptr + req_pool_idx * acc_stride_req + head * acc_stride_head
        # In CEIL numbering is_partial implies the block strictly overruns the
        # request end, so remainder is in [1, BLOCK_N) — always > 0.  (The old
        # "+bsz" layout could schedule an empty trailing row with remainder == 0,
        # which needed guarding; that case no longer exists.)
        tl.store(base + offs_d, kmax.to(acc_ptr.dtype.element_ty), mask=d_mask)
        tl.store(base + D + offs_d, kmin.to(acc_ptr.dtype.element_ty), mask=d_mask)
        tl.store(base + 2 * D + offs_d, ksum.to(acc_ptr.dtype.element_ty), mask=d_mask)
        # This block's `pooled` row is unused (the norm/rope kernel skips partial
        # blocks); the host allocates `pooled` with torch.zeros, so the GEMM reads
        # a clean 0 here — no in-kernel zeroing needed.
    else:
        # complete block -> pooled [max | min | avg] (feeds the GEMM).
        kmax = tl.max(k, axis=0)
        kmin = tl.min(k, axis=0)
        kavg = tl.sum(k, axis=0) / BLOCK_N
        base = pooled_ptr + blk * p_stride_blk + head * p_stride_head
        tl.store(base + offs_d, kmax, mask=d_mask)
        tl.store(base + D + offs_d, kmin, mask=d_mask)
        tl.store(base + 2 * D + offs_d, kavg, mask=d_mask)


@triton.jit
def _prefill_normrope_kernel(
    proj_ptr,  # [total_blocks, Hk, GATE_D] fp32  projected (pre-norm) compressed-K
    pr_stride_blk,
    pr_stride_head,
    cum_block_cnt_ptr,  # [bsz+1] int32
    cu_seqlens_ptr,  # [bsz+1] int32
    req_pool_indices_ptr,  # [bsz] int32
    prefix_blocks_ptr,  # [bsz] int32  per-request global block offset (chunked prefill)
    req_to_summary_ptr,  # [max_num_reqs, max_blocks] block -> summary slot
    rts_stride_req,
    rts_stride_blk,
    norm_ptr,  # [GATE_D] knorm weight
    inv_freq_ptr,  # [HALF] rope inverse frequencies (fp32)
    summary_ptr,  # [summary_pool, Hk, GATE_D] output
    s_stride_slot,
    s_stride_head,
    eps,
    BSZ: tl.constexpr,
    BSZ_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GATE_D: tl.constexpr,
    HALF: tl.constexpr,
    HALF_POW2: tl.constexpr,
):
    """RMSNorm + NeoX RoPE the projected block -> its summary slot.

    Grid ``(total_blocks, Hk)``.  Each program locates itself (``_locate_block``);
    trailing partial blocks (no complete summary) early-return.  The summary slot
    is looked up in-kernel from the recovered ``(req_pool_index, GLOBAL block id)`` —
    no host-side gather.  Numerics identical to the norm+rope half of
    ``_prefill_summary_kernel`` (shared ``_norm_and_rope`` device fn).

    CHUNKED PREFILL: ``_locate_block`` returns the block id WITHIN THIS CHUNK
    (``intra_req_blk_id``); the request-GLOBAL block id is
    ``prefix_blocks[batch_id] + intra_req_blk_id``.  Both the RoPE position
    (``global_blk_id*BLOCK_N``) and the ``req_to_summary`` slot lookup use the
    global id, so a chunk starting mid-request writes to the correct positions /
    slots.  For a fresh single-chunk prefill ``prefix_blocks`` is 0 -> identical to
    the old ``intra_req_blk_id``-based math.
    """
    blk = tl.program_id(0)
    head = tl.program_id(1)

    batch_id, blk_tok_start, intra_req_blk_id, remainder, is_partial = _locate_block(
        blk, cum_block_cnt_ptr, cu_seqlens_ptr, BSZ, BSZ_POW2, BLOCK_N
    )
    if is_partial:
        return  # partial blocks have no complete-block summary

    # Global (request-relative) block id: prefix chunks already summarized
    # `prefix_blocks[batch_id]` blocks before this chunk began.
    pb = tl.load(prefix_blocks_ptr + batch_id).to(tl.int32)
    global_blk_id = pb + intra_req_blk_id
    pos = global_blk_id * BLOCK_N  # RoPE position = block-start position within request
    req_pool_idx = tl.load(req_pool_indices_ptr + batch_id).to(tl.int64)
    slot = tl.load(
        req_to_summary_ptr + req_pool_idx * rts_stride_req + global_blk_id * rts_stride_blk
    ).to(tl.int64)

    offs_h = tl.arange(0, HALF_POW2)
    h_mask = offs_h < HALF
    src = proj_ptr + blk * pr_stride_blk + head * pr_stride_head
    o1 = tl.load(src + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    o2 = tl.load(src + offs_h + HALF, mask=h_mask, other=0.0).to(tl.float32)

    out_lo, out_hi = _norm_and_rope(
        o1, o2, norm_ptr, inv_freq_ptr, pos, eps,
        GATE_D=GATE_D, HALF=HALF, HALF_POW2=HALF_POW2,
    )

    dst = summary_ptr + slot * s_stride_slot + head * s_stride_head
    tl.store(dst + offs_h, out_lo.to(summary_ptr.dtype.element_ty), mask=h_mask)
    tl.store(dst + offs_h + HALF, out_hi.to(summary_ptr.dtype.element_ty), mask=h_mask)


# ---------------------------------------------------------------------------
# Kernel C: decode-step rolling-fold + (on block fill) summary update
# (graph-safe, grid (bsz, kv head))
# ---------------------------------------------------------------------------
@triton.jit
def _decode_summary_kernel(
    k_ptr,  # [bsz, Hk, D] new token's raw key (pre-RoPE, post-k_norm)
    k_stride_tok,
    k_stride_head,
    acc_ptr,  # [max_num_reqs, Hk, 3*D] fp32  rolling [max | min | sum]
    acc_stride_req,
    acc_stride_head,
    seq_lens_ptr,  # [bsz] int32/int64  includes the current token
    req_pool_indices_ptr,  # [bsz] int64
    req_to_summary_ptr,  # [max_num_reqs, max_blocks]
    rts_stride_req,
    rts_stride_blk,
    w_ptr,  # [Hk, 3*D, GATE_D] gate K-branch weight
    w_stride_head,
    w_stride_c,
    w_stride_g,
    norm_ptr,  # [GATE_D] knorm weight
    inv_freq_ptr,  # [HALF] rope inverse frequencies (fp32)
    summary_ptr,  # [summary_pool, Hk, GATE_D] output
    s_stride_slot,
    s_stride_head,
    eps,
    BLOCK_N: tl.constexpr,  # == block_size
    D: tl.constexpr,
    GATE_D: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HALF_POW2: tl.constexpr,
):
    req = tl.program_id(0)
    head = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)
    # Skip CUDA-graph padding rows.  Under graph replay a real batch is padded
    # up to the captured bs; padding rows carry seq_len == fill_value (1) and
    # req_pool_indices == 0 (the zero-init buffer is only overwritten for the
    # raw_bs real rows).  A real decoding request always has seq_len >= 2
    # (>=1 prefilled token + the current one), so seq_len > 1 cleanly excludes
    # padding.  Without this guard a padding row would fold its garbage key into
    # acc[req_pool_idx=0] and race the real request that owns pool slot 0.
    # (A data-dependent branch *inside* a kernel is CUDA-graph-safe; capture
    # records only the fixed (bsz, Hk) launch.)
    if seq_len > 1:
        req_pool_idx = tl.load(req_pool_indices_ptr + req).to(tl.int64)
        pos = seq_len - 1  # current-token position
        is_first = (pos % BLOCK_N) == 0  # first token of a fresh block

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D

        # --- fold the new token's key into the rolling [max | min | sum] acc ---
        # (self-resetting on the first token of a block).  Padded d-lanes load /
        # store as 0 (knew/prev = 0 there), so max|min|sum stay 0 -> kavg 0,
        # matching _compress_and_rope's "padded lanes = 0" contract.
        knew = tl.load(
            k_ptr + req * k_stride_tok + head * k_stride_head + offs_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32) # shape: (D,)
        base = acc_ptr + req_pool_idx * acc_stride_req + head * acc_stride_head
        max_ptr = base + offs_d  # [0, D)
        min_ptr = base + D + offs_d  # [D, 2D)
        sum_ptr = base + 2 * D + offs_d  # [2D, 3D)
        prev_max = tl.load(max_ptr, mask=d_mask, other=0.0)
        prev_min = tl.load(min_ptr, mask=d_mask, other=0.0)
        prev_sum = tl.load(sum_ptr, mask=d_mask, other=0.0)
        new_max = tl.where(is_first, knew, tl.maximum(prev_max, knew))
        new_min = tl.where(is_first, knew, tl.minimum(prev_min, knew))
        new_sum = tl.where(is_first, knew, prev_sum + knew)
        tl.store(max_ptr, new_max, mask=d_mask)
        tl.store(min_ptr, new_min, mask=d_mask)
        tl.store(sum_ptr, new_sum, mask=d_mask)

        # --- on block fill, compress the just-completed block -> summary ---
        # The block just filled iff seq_len % block_size == 0; the fresh
        # new_max/new_min/new_sum are still in registers, so we feed them
        # straight into the compression with no re-read of acc.  avg =
        # sum / block_size (a block that fills during decode always holds
        # exactly BLOCK_N tokens).
        if seq_len % BLOCK_N == 0:
            block_id = pos // BLOCK_N
            slot = tl.load(
                req_to_summary_ptr
                + req_pool_idx * rts_stride_req
                + block_id * rts_stride_blk
            ).to(tl.int64)
            kavg = new_sum / BLOCK_N

            out_lo, out_hi = _compress_and_rope(
                new_max,
                new_min,
                kavg,
                w_ptr,
                head * w_stride_head,
                w_stride_c,
                w_stride_g,
                norm_ptr,
                inv_freq_ptr,
                pos,
                eps,
                D=D,
                GATE_D=GATE_D,
                HALF=HALF,
                BLOCK_D=BLOCK_D,
                HALF_POW2=HALF_POW2,
            )

            offs_h = tl.arange(0, HALF_POW2)
            h_mask = offs_h < HALF
            dst = summary_ptr + slot * s_stride_slot + head * s_stride_head
            tl.store(dst + offs_h, out_lo.to(summary_ptr.dtype.element_ty), mask=h_mask)
            tl.store(
                dst + offs_h + HALF,
                out_hi.to(summary_ptr.dtype.element_ty),
                mask=h_mask,
            )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@dataclass
class PrefillSummarySchedule:
    """Layer-independent host schedule for the prefill summary/rolling build.

    Minimal: just the per-request block/token boundaries.  The kernels recover
    each block's ``(batch_id, intra-request block id, is_partial)`` on the fly by
    locating its global block id in ``cum_block_cnt`` — so the host no longer
    flattens per-block ``[total_blocks]`` arrays (block_tok_start / block_pos /
    summary_slot) via repeat_interleave + gather.  Every field depends only on
    ``extend_seq_lens`` / ``req_pool_indices`` / ``block_size`` (layer-independent),
    so this is built once per forward and reused across all seer layers.

    Block numbering is CEIL: request ``i`` owns ``ceil(seq_len_i/block_size)``
    blocks, the last of which is the trailing (possibly partial) block.  A single
    grid axis ``total_blocks`` covers both complete blocks (-> pooled, for the
    summary GEMM) and trailing partial blocks (-> rolling accumulator seed); the
    kernel tells them apart by whether the block's token-end exceeds the request's
    token-end (``cu_seqlens[batch_id+1]``).
    """

    bsz: int
    total_blocks: int  # sum of ceil(seq_len/block_size) over requests (host int)
    # int32 is sufficient: cu_seqlens holds the per-forward token cumsum (bounded
    # by max_total_tokens, well under 2^31) and cum_block_cnt is even smaller.  The
    # pool kernel promotes blk_tok_start to int64 before multiplying by the K-row
    # stride, so token addressing never overflows even at extreme context.
    cum_block_cnt: torch.Tensor  # [bsz+1] int32  exclusive cumsum of per-req block count
    cu_seqlens: torch.Tensor  # [bsz+1] int32  exclusive cumsum of extend_seq_lens (token starts)
    rpi32: torch.Tensor  # [bsz] int32  req_pool_indices (for summary_slot lookup)
    # [bsz] int32  per-request prefix block offset = extend_prefix_lens//block_size.
    # For a fresh (non-chunked) prefill this is all-zeros; for a CHUNKED prefill it
    # is the number of blocks already summarized by earlier chunks, so the norm/rope
    # kernel can turn each intra-chunk block id into the request-GLOBAL block id
    # (prefix_blocks[batch_id] + intra_req_blk_id) it needs for both the RoPE
    # position and the req_to_summary slot lookup.  REQUIRES each chunk boundary to
    # be block_size-aligned (asserted in build_prefill_summary_schedule) so a full
    # block never straddles two chunks.
    prefix_blocks: torch.Tensor  # [bsz] int32  extend_prefix_lens // block_size


def build_prefill_summary_schedule(
    extend_seq_lens: torch.Tensor,  # [bsz] prefill length of each request (device)
    req_pool_indices: torch.Tensor,  # [bsz] (device)
    block_size: int,
    extend_seq_lens_cpu: torch.Tensor | list,  # [bsz] host copy (required, sync-free)
    extend_prefix_lens: torch.Tensor | None = None,  # [bsz] tokens already cached (device)
    extend_prefix_lens_cpu: torch.Tensor | list | None = None,  # [bsz] host copy
) -> PrefillSummarySchedule:
    """Build the per-forward (layer-independent) prefill summary/rolling schedule.

    Produces only per-request boundaries (no per-block flattening): the exclusive
    cumsums of the per-request block count (``cum_block_cnt``, CEIL semantics) and
    of the token count (``cu_seqlens``).  The kernels derive everything per-block
    from these via an in-kernel search (see ``_locate_block``).

    ``total_blocks`` (the grid size) is a host int derived from the required
    host-side ``extend_seq_lens_cpu`` (``forward_batch.extend_seq_lens_cpu``, always
    populated by the scheduler) — NO device sync.  Block summary slots are NOT
    gathered here; the norm/rope kernel looks up each block's slot directly from
    ``req_to_summary`` with its recovered ``(req_pool_index, GLOBAL block id)``.

    Safe to call in ``init_forward_metadata``: extend is never CUDA-graph captured.

    CHUNKED PREFILL: ``extend_prefix_lens`` gives the tokens ALREADY summarized by
    earlier chunks of each request.  ``prefix_blocks = extend_prefix_lens //
    block_size`` is the request-global block offset the norm/rope kernel adds to the
    intra-chunk block id, so a chunk that starts mid-request writes its blocks to the
    correct global slots / RoPE positions.  The chunk boundary MUST be
    block_size-aligned (asserted below) so a complete block never straddles two
    chunks — the pool/rolling kernels operate purely on the current chunk's tokens
    and stay correct without change.  For a fresh single-chunk prefill from position
    0 (``extend_prefix_lens`` is None / all-zeros) this reduces exactly to the old
    behavior (``prefix_blocks == 0``).
    """
    device = extend_seq_lens.device
    ext = extend_seq_lens.to(device=device, dtype=torch.int64)
    bsz = ext.shape[0]

    # Per-request CEIL block count (trailing partial counted as one block).
    block_cnt = (ext + block_size - 1) // block_size  # [bsz]
    cum_block_cnt = torch.zeros(bsz + 1, dtype=torch.int32, device=device)
    cum_block_cnt[1:] = block_cnt.cumsum(0).to(torch.int32)

    # Token starts: exclusive cumsum of extend_seq_lens.  (forward_batch.
    # extend_start_loc is the same [bsz] exclusive cumsum, but lacks the trailing
    # total-token sentinel _locate_block needs as each request's token-end; reusing
    # it would save one tiny [bsz] cumsum at the cost of threading it through plus
    # a kernel-side "last request -> num_tokens" branch — not worth it off the
    # hot path, so we recompute the self-contained [bsz+1] form here.)
    cu_seqlens = torch.zeros(bsz + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = ext.cumsum(0).to(torch.int32)

    rpi32 = req_pool_indices.to(device=device, dtype=torch.int32)

    # Per-request prefix block offset for CHUNKED prefill.  Without a prefix (fresh
    # single-chunk prefill from 0) this is all-zeros and the kernel math is a no-op.
    if extend_prefix_lens is None:
        prefix_blocks = torch.zeros(bsz, dtype=torch.int32, device=device)
    else:
        # Each chunk boundary must be block_size-aligned so a complete block never
        # straddles chunks.  Assert on the host copy (no device sync); require it to
        # be provided when a device prefix is (the backend always passes both).
        assert extend_prefix_lens_cpu is not None, (
            "extend_prefix_lens_cpu must accompany extend_prefix_lens (host copy, "
            "so the block_size-alignment check needs no device sync)"
        )
        if isinstance(extend_prefix_lens_cpu, torch.Tensor):
            _pref_cpu = extend_prefix_lens_cpu.tolist()
        else:
            _pref_cpu = list(extend_prefix_lens_cpu)
        misaligned = [int(p) for p in _pref_cpu if int(p) % block_size != 0]
        assert not misaligned, (
            f"chunked-prefill requires every chunk boundary aligned to block_size="
            f"{block_size}; got prefix lengths {misaligned} that are not multiples "
            f"of {block_size}.  Set chunked_prefill_size to a multiple of "
            f"{block_size} (the seer summary/rolling build assumes complete blocks "
            f"never straddle a chunk)."
        )
        prefix_blocks = (
            extend_prefix_lens.to(device=device, dtype=torch.int64) // block_size
        ).to(torch.int32)

    # Host scalar grid size with NO device sync: derive total_blocks from the host
    # copy of the lengths.  Its concrete type is genuinely not fixed — the normal
    # scheduler extend path sets forward_batch.extend_seq_lens_cpu to a python
    # list[int] (from ModelWorkerBatch.extend_seq_lens), while the fake-extend /
    # CUDA-graph-capture path (and our tests) set it to a CPU torch.Tensor via
    # .cpu().  Both are already on the host, so normalize tensor -> list once and
    # take a single arithmetic path; either way no device sync.  We deliberately
    # do NOT fall back to a device .item() — a caller without the host copy is a
    # bug we'd rather surface than silently pay a sync for.
    assert extend_seq_lens_cpu is not None, (
        "extend_seq_lens_cpu must be provided (forward_batch.extend_seq_lens_cpu) "
        "so total_blocks needs no device sync"
    )
    if isinstance(extend_seq_lens_cpu, torch.Tensor):
        extend_seq_lens_cpu = extend_seq_lens_cpu.tolist()
    total_blocks = int(
        sum((int(s) + block_size - 1) // block_size for s in extend_seq_lens_cpu)
    )

    return PrefillSummarySchedule(
        bsz=bsz,
        total_blocks=total_blocks,
        cum_block_cnt=cum_block_cnt,
        cu_seqlens=cu_seqlens,
        rpi32=rpi32,
        prefix_blocks=prefix_blocks,
    )


def update_summary_cache_prefill(
    schedule: PrefillSummarySchedule,  # layer-independent, from build_*_schedule
    k_nope: torch.Tensor,  # [num_tokens, Hk, D] pre-RoPE post-k_norm keys
    summary_cache: torch.Tensor,  # [summary_pool, Hk, GATE_D] (written in place)
    rolling_buffer: torch.Tensor,  # [max_num_reqs, Hk, 3*D] fp32 [max|min|sum] (in place)
    req_to_summary: torch.Tensor,  # [max_num_reqs, max_blocks] block -> summary slot
    proj_weight: torch.Tensor,  # [Hk, D*3, GATE_D] attngate_linear_k.weight
    norm_weight: torch.Tensor,  # [GATE_D] attngate_knorm.weight
    rope_inv_freqs: torch.Tensor,  # [GATE_D//2] fp32 NeoX inverse frequencies
    block_size: int,
    eps: float = 1e-6,
) -> None:
    """Seed both prefill caches in place: per-complete-block summaries and the
    trailing-partial-block rolling accumulator, using a precomputed (per-forward,
    layer-independent) :class:`PrefillSummarySchedule`.

    Split path (pool -> cuBLAS GEMM -> norm/RoPE), single ``total_blocks`` grid
    (CEIL); each program locates its ``(request, intra-block id, is_partial)`` from
    the schedule's per-request boundaries (see :func:`_locate_block`):

    * :func:`_prefill_pool_kernel` (grid ``(total_blocks, Hk)``): complete blocks
      pool ``[max|min|avg]`` into ``pooled``; each request's trailing partial block
      pools ``[max|min|sum]`` into the rolling accumulator (subsumes the old
      ``_seed_rolling_kernel``).
    * batched GEMM ``pooled @ proj_weight`` (cuBLAS): the bandwidth-heavy linear_k,
      weight loaded once and reused across all blocks.
    * :func:`_prefill_normrope_kernel` (grid ``(total_blocks, Hk)``): RMSNorm +
      NeoX RoPE at the block-start position, written to each block's summary slot
      (looked up in-kernel; partial blocks early-return).

    Only the per-layer kernel launches happen here; all host bookkeeping (no
    device sync; see :func:`build_prefill_summary_schedule`) is done once.
    """
    num_tokens, Hk, D = k_nope.shape
    GATE_D = summary_cache.shape[-1]
    HALF = GATE_D // 2
    BLOCK_D = triton.next_power_of_2(D)
    HALF_POW2 = triton.next_power_of_2(HALF)
    bsz = schedule.bsz
    total_blocks = schedule.total_blocks
    BSZ_POW2 = triton.next_power_of_2(bsz)

    if total_blocks == 0:
        # Unreachable in practice: every extend request has >= 1 prefill token, so
        # CEIL block count >= 1 and total_blocks >= bsz >= 1.  Kept as a cheap O(1)
        # guard for a degenerate 0-request / empty-extend batch (grid (0,Hk) and a
        # 0-row GEMM would otherwise run as harmless no-ops, but this is clearer).
        return

    # --- Pool: complete blocks -> `pooled` (for the GEMM); each request's
    # trailing partial block -> rolling accumulator.  Single grid over all
    # `total_blocks` (CEIL); _locate_block tells the two apart.  `pooled` is
    # zero-initialized so partial-block rows (which the pool kernel skips, and the
    # norm/rope kernel ignores) feed a clean 0 through the GEMM — no NaN from
    # uninitialized memory, and no in-kernel zeroing needed.
    pooled = torch.zeros(
        (total_blocks, Hk, _K_POOL_DUP * D), dtype=torch.float32, device=k_nope.device
    )
    _prefill_pool_kernel[(total_blocks, Hk)](
        k_nope,
        k_nope.stride(0),
        k_nope.stride(1),
        schedule.cum_block_cnt,
        schedule.cu_seqlens,
        schedule.rpi32,
        pooled,
        pooled.stride(0),
        pooled.stride(1),
        rolling_buffer,
        rolling_buffer.stride(0),
        rolling_buffer.stride(1),
        BSZ=bsz,
        BSZ_POW2=BSZ_POW2,
        BLOCK_N=block_size,
        D=D,
        BLOCK_D=BLOCK_D,
    )

    # Projection: [total_blocks, Hk, 3D] x [Hk, 3D, GATE_D] -> [total_blocks, Hk, GATE_D].
    # bmm over the head axis (cuBLAS batched GEMM); cast weight to fp32 to match
    # the fused kernel's fp32 accumulation.
    proj = torch.einsum(
        "nhc,hcg->nhg", pooled, proj_weight.to(torch.float32)
    ).contiguous()

    _prefill_normrope_kernel[(total_blocks, Hk)](
        proj,
        proj.stride(0),
        proj.stride(1),
        schedule.cum_block_cnt,
        schedule.cu_seqlens,
        schedule.rpi32,
        schedule.prefix_blocks,
        req_to_summary,
        req_to_summary.stride(0),
        req_to_summary.stride(1),
        norm_weight,
        rope_inv_freqs,
        summary_cache,
        summary_cache.stride(0),
        summary_cache.stride(1),
        eps,
        BSZ=bsz,
        BSZ_POW2=BSZ_POW2,
        BLOCK_N=block_size,
        GATE_D=GATE_D,
        HALF=HALF,
        HALF_POW2=HALF_POW2,
    )


def update_summary_cache_decode(
    k: torch.Tensor,  # [bsz, Hk, D] new token's raw key (pre-RoPE, post-k_norm)
    rolling_buffer: torch.Tensor,  # [max_num_reqs, Hk, 3*D] fp32 [max|min|sum]
    seq_lens: torch.Tensor,  # [bsz] includes the current token (device)
    req_pool_indices: torch.Tensor,  # [bsz] (device)
    req_to_summary: torch.Tensor,  # [max_num_reqs, max_blocks] block -> summary slot
    summary_cache: torch.Tensor,  # [summary_pool, Hk, GATE_D] (written in place)
    proj_weight: torch.Tensor,  # [Hk, D*3, GATE_D] attngate_linear_k.weight
    norm_weight: torch.Tensor,  # [GATE_D] attngate_knorm.weight
    rope_inv_freqs: torch.Tensor,  # [GATE_D//2] fp32 NeoX inverse frequencies
    block_size: int,
    norm_eps: float = 1e-6,
) -> None:
    """Decode-step rolling-fold + (on block fill) summary compression.

    Fused per decode token per request: folds the new token's raw K ``k`` into
    the rolling ``[max | min | sum]`` accumulator (self-resetting on the first
    token of a block), then — when the request's block just filled
    (``seq_len % block_size == 0``) — recovers ``avg = sum / block_size`` from
    the freshly-updated reductions (still in registers, no re-read of the
    accumulator), projects (gate K-branch), RMSNorms, applies NeoX RoPE at the
    *completing-token* position (``seq_len - 1``), and writes the result to
    summary slot ``req_to_summary[req, (seq_len-1)//block_size]``.

    Graph-safe by construction: fixed grid ``(bsz, Hk)``, no host-side syncs /
    branches / dynamic shapes.  The ``seq_len > 1`` (padding) and
    ``seq_len % block_size == 0`` (fill) tests live *inside* the kernel (a
    data-dependent branch within a kernel does not break CUDA-graph capture,
    which records only the launch); padding rows and non-filling requests store
    no summary.
    """
    bsz, Hk, D = k.shape
    GATE_D = summary_cache.shape[-1]
    HALF = GATE_D // 2
    BLOCK_D = triton.next_power_of_2(D)
    HALF_POW2 = triton.next_power_of_2(HALF)

    # update rolling buffer -> (project -> norm -> RoPE -> write summary, if needed)
    # NOTE (evaluated, NOT splitting — keep this fused single kernel): splitting
    # the projection into a separate GEMM "to save projection-weight bandwidth"
    # (as done for prefill) does NOT pay off in the decode regime, for four
    # reasons specific to decode:
    #   1. A block fills only when seq_len % block_size == 0 — on average once per
    #      block_size (e.g. 64) steps per request — so this kernel touches the
    #      projection weight only inside that rare branch; most steps never load
    #      it.  There is little weight-bandwidth to save in the first place.
    #   2. This path is CUDA-graph captured.  "Project only the rows that fill
    #      this step" is data-dependent (the fill count varies per step), but a
    #      graph-captured GEMM must have a fixed shape, so it would have to
    #      project ALL bsz rows unconditionally every step — turning a ~1/block_size
    #      matvec into an every-step GEMM.  At small bsz that is a net loss.
    #   3. Even at large bsz the expected fills/step are few (~bsz/block_size), so
    #      a GEMM over those rows has almost no weight reuse to exploit.
    #   4. The current fusion already feeds the freshly-folded max/min/sum from
    #      registers straight into the compression on fill (no acc re-read), which
    #      is exactly the right shape for decode; splitting would add intermediate
    #      materialization + extra launches on an already launch-bound decode step.
    _decode_summary_kernel[(bsz, Hk)](
        k,
        k.stride(0),
        k.stride(1),
        rolling_buffer,
        rolling_buffer.stride(0),
        rolling_buffer.stride(1),
        seq_lens,
        req_pool_indices,
        req_to_summary,
        req_to_summary.stride(0),
        req_to_summary.stride(1),
        proj_weight,
        proj_weight.stride(0),
        proj_weight.stride(1),
        proj_weight.stride(2),
        norm_weight,
        rope_inv_freqs,
        summary_cache,
        summary_cache.stride(0),
        summary_cache.stride(1),
        norm_eps,
        BLOCK_N=block_size,
        D=D,
        GATE_D=GATE_D,
        HALF=HALF,
        BLOCK_D=BLOCK_D,
        HALF_POW2=HALF_POW2,
    )