from __future__ import annotations

"""Triton kernels and launchers for block-sparse key-summary cache maintenance.

The block-sparse (MoBA-style) backend keeps a *summary* (mean of the keys) of
every **complete** logical KV block.  A block's summary is defined once and for
all as the mean of its ``block_size`` keys read from the paged ``k_buffer`` — the
*same* definition everywhere (prefill recompute, extend, decode write-on-fill),
so all three paths are bit-identical.

The trailing **partial** block is never summarised: routing only ever scores
*complete previous* blocks (the current/partial block is appended by position,
never scored — see
:func:`~sglang.srt.layers.attention.blocksparse.common_index_kernels.block_inds_to_kv_inds_for_decoding`).
A partial block gets its summary lazily, the moment it fills during decode
(:func:`update_block_summary_decode`).  This removes the old rolling-mean merge
entirely and makes chunked prefill trivially correct (a block straddling two
chunks is simply skipped until complete, then read whole from the cache).

One kernel, :func:`_build_block_summaries_from_cache_kernel`, builds a contiguous
range of complete blocks ``[start_block, end_block)`` per request from the cache.
Two launchers select the range and the key source:

* :func:`recompute_all_prefix_summaries` — prefix blocks ``[0, prefix//bs)`` for
  *all* layers (the prefix KV is already cached up-front), run once in
  ``init_forward_metadata``.
* :func:`update_block_summary` — the *new complete* blocks of this extend,
  ``[prefix//bs, end//bs)``, per layer (the new keys are written to ``k_buffer``
  by ``set_kv_buffer`` before the per-layer summary build runs).

The launcher functions are pure (no backend ``self`` state); the backend passes
its config explicitly.
"""

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool import BlockSparseTokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# ---------------------------------------------------------------------------
#  Triton kernel: build complete-block summaries from the paged KV cache
#
#  Grid (bsz, max_blocks, H): one program per (request, relative-block, kv-head).
#  Each program builds ONE complete block's summary = mean of its block_size keys
#  read from k_buffer via req_to_token, written straight to the summary slot.  No
#  rolling merge, no old-summary read: a block's summary is a pure function of its
#  cached keys, identical across prefill / extend / decode.
#
#  The per-request block range is [start_block, start_block + n_blocks):
#    * prefix recompute: start_block=0,                n_blocks=prefix_len//bs
#    * extend new blocks: start_block=prefix_len//bs,  n_blocks=(complete blocks
#                         added by this extend)
#  Programs with pid_blk_rel >= n_blocks early-return (grid is over-allocated to
#  the batch max).  Trailing partial blocks are never built here (lazy: filled
#  during decode).
# ---------------------------------------------------------------------------


@triton.jit
def _build_block_summaries_from_cache_kernel(
    # -- KV cache key buffer (for one layer) --
    k_buffer_ptr,  # [pool_size, H, D]
    k_buffer_stride_tok,
    k_buffer_stride_head,
    # -- req_to_token: cached token locations --
    req_to_token_ptr,  # [req_pool_size, max_ctx_len]  int32
    req_to_token_stride,
    # -- req_to_summary: block -> summary slot --
    req_to_summary_ptr,  # [req_pool_size, max_num_summary]  int32
    req_to_summary_stride,
    # -- per-request info --
    req_pool_indices_ptr,  # [bsz]  int64
    start_block_ptr,  # [bsz]  int32 – first block this launch builds for the req
    n_blocks_ptr,  # [bsz]  int32 – number of complete blocks to build
    # -- summary buffer (for one layer) --
    summary_buf_ptr,  # [summary_pool_size+1, H, D]
    summary_buf_stride_slot,
    summary_buf_stride_head,
    # -- constants --
    block_size: tl.constexpr,
    D: tl.constexpr,
    D_BLOCK: tl.constexpr,  # next_power_of_2(D), for masking
    BLOCK_N: tl.constexpr,  # next_power_of_2(block_size), token-axis tile
):
    pid_req = tl.program_id(0)
    pid_blk_rel = tl.program_id(1)
    pid_head = tl.program_id(2)

    n_blocks = tl.load(n_blocks_ptr + pid_req).to(tl.int32)
    # Grid is over-allocated to the batch-max block count; skip the tail.
    if pid_blk_rel >= n_blocks:
        return

    start_block = tl.load(start_block_ptr + pid_req).to(tl.int32)
    abs_block = start_block + pid_blk_rel
    req_pool_idx = tl.load(req_pool_indices_ptr + pid_req).to(tl.int64)

    summary_loc = tl.load(
        req_to_summary_ptr + req_pool_idx * req_to_summary_stride + abs_block
    ).to(tl.int32)
    # Slots for complete blocks are preallocated; guard defensively (slot 0 is the
    # padding sentinel).
    if summary_loc <= 0:
        return

    blk_start = abs_block * block_size
    d_offsets = tl.arange(0, D_BLOCK)
    d_mask = d_offsets < D
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < block_size  # mask the token-tile tail when bs not pow2

    # Vectorized: load all block_size token locations, do one 2D gather
    # [BLOCK_N, D_BLOCK] from the (scattered) paged cache, reduce over the token
    # axis.  Replaces the old per-token serial loop.
    tok_locs = tl.load(
        req_to_token_ptr + req_pool_idx * req_to_token_stride + blk_start + offs_n,
        mask=n_mask,
        other=0,
    ).to(tl.int64)
    k = tl.load(
        k_buffer_ptr
        + tok_locs[:, None] * k_buffer_stride_tok
        + pid_head * k_buffer_stride_head
        + d_offsets[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(k, axis=0) / block_size

    s_ptr = (
        summary_buf_ptr
        + summary_loc * summary_buf_stride_slot
        + pid_head * summary_buf_stride_head
        + d_offsets
    )
    tl.store(s_ptr, mean.to(summary_buf_ptr.dtype.element_ty), mask=d_mask)


# ---------------------------------------------------------------------------
#  Launchers (pure functions; backend passes its config explicitly)
# ---------------------------------------------------------------------------


def update_block_summary(
    layer: "RadixAttention",
    forward_batch: "ForwardBatch",
    req_to_token: torch.Tensor,
    req_to_summary: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Build summaries for the *new complete* blocks added by this extend (lazy).

    A block's summary is the mean of its ``block_size`` keys read straight from
    the paged ``k_buffer`` via ``req_to_token`` — the same definition used by the
    prefix recompute and the decode write-on-fill, so all three are bit-identical
    (no rolling merge).  This layer's new keys are already in ``k_buffer``
    (``set_kv_buffer`` runs before this build in ``forward_extend``).

    Per request the built range is the complete blocks ``[prefix//bs, end//bs)``
    newly completed by this extend; the *trailing partial* block is deliberately
    NOT built — its summary is never scored until it fills during decode, where
    :func:`update_block_summary_decode` builds it.  This also makes chunked
    prefill correct for free (a partial block straddling chunks is skipped until
    complete).

    Grid bound (``max_blocks``) is taken from the host-side
    ``extend_prefix_lens_cpu`` / ``extend_seq_lens_cpu`` — no device sync.
    """
    token_to_kv_pool: "BlockSparseTokenToKVPool" = forward_batch.token_to_kv_pool
    device = forward_batch.req_pool_indices.device

    # Per-request block ranges, host-side (List[int]) -> no device sync.
    prefix_cpu = forward_batch.extend_prefix_lens_cpu
    ext_cpu = forward_batch.extend_seq_lens_cpu
    assert prefix_cpu is not None and ext_cpu is not None, (
        "update_block_summary needs extend_{prefix,seq}_lens_cpu (host copies) "
        "to size the grid without a device sync"
    )
    start_list = [int(p) // block_size for p in prefix_cpu]
    # End of complete blocks = (prefix + extend) // block_size; the trailing
    # partial block (if any) is excluded.
    end_list = [
        (int(p) + int(e)) // block_size for p, e in zip(prefix_cpu, ext_cpu)
    ]
    n_list = [max(0, end - st) for st, end in zip(start_list, end_list)]
    max_blocks = max(n_list) if n_list else 0
    if max_blocks <= 0:  # no block completed this extend
        return

    bsz = forward_batch.req_pool_indices.size(0)
    start_block = torch.tensor(start_list, dtype=torch.int32, device=device)
    n_blocks = torch.tensor(n_list, dtype=torch.int32, device=device)
    summary_buf = token_to_kv_pool.get_summary_buffer(layer.layer_id)
    k_buffer = token_to_kv_pool.get_key_buffer(layer.layer_id)
    D_BLOCK = triton.next_power_of_2(head_dim)
    BLOCK_N = triton.next_power_of_2(block_size)
    _build_block_summaries_from_cache_kernel[(bsz, max_blocks, num_kv_heads)](
        k_buffer,
        k_buffer.stride(0),
        k_buffer.stride(1),
        req_to_token,
        req_to_token.stride(0),
        req_to_summary,
        req_to_summary.stride(0),
        forward_batch.req_pool_indices,
        start_block,
        n_blocks,
        summary_buf,
        summary_buf.stride(0),
        summary_buf.stride(1),
        block_size=block_size,
        D=head_dim,
        D_BLOCK=D_BLOCK,
        BLOCK_N=BLOCK_N,
    )


def recompute_all_prefix_summaries(
    forward_batch: "ForwardBatch",
    req_to_token: torch.Tensor,
    req_to_summary: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Build prefix-block summaries for ALL layers in one pass (range [0, prefix//bs)).

    Called once from ``init_forward_metadata`` (the prefix KV is cached up-front
    for every layer, so this is layer-independent setup done before the per-layer
    extend builds).  Uses the shared complete-block builder; grid bound comes from
    the host-side ``extend_prefix_lens_cpu`` — no device sync.
    """
    prefix_cpu = forward_batch.extend_prefix_lens_cpu
    assert prefix_cpu is not None, (
        "recompute_all_prefix_summaries needs extend_prefix_lens_cpu (host copy)"
    )
    n_list = [int(p) // block_size for p in prefix_cpu]
    max_prefix_blocks = max(n_list) if n_list else 0
    if max_prefix_blocks <= 0:  # no complete prefix blocks
        return

    device = forward_batch.req_pool_indices.device
    bsz = forward_batch.req_pool_indices.size(0)
    n_blocks = torch.tensor(n_list, dtype=torch.int32, device=device)
    # Prefix blocks always start at block 0.
    start_block = torch.zeros_like(n_blocks)

    token_to_kv_pool: "BlockSparseTokenToKVPool" = forward_batch.token_to_kv_pool
    req_pool_indices = forward_batch.req_pool_indices
    start_layer = token_to_kv_pool.start_layer
    num_layers = token_to_kv_pool.layer_num
    D_BLOCK = triton.next_power_of_2(head_dim)
    BLOCK_N = triton.next_power_of_2(block_size)
    # TODO: the loop can be fused into a CUDA C kernel to achieve single kernel launch
    for layer_id in range(start_layer, start_layer + num_layers):
        k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
        summary_buf = token_to_kv_pool.get_summary_buffer(layer_id)
        _build_block_summaries_from_cache_kernel[
            (bsz, max_prefix_blocks, num_kv_heads)
        ](
            k_buffer,
            k_buffer.stride(0),
            k_buffer.stride(1),
            req_to_token,
            req_to_token.stride(0),
            req_to_summary,
            req_to_summary.stride(0),
            req_pool_indices,
            start_block,
            n_blocks,
            summary_buf,
            summary_buf.stride(0),
            summary_buf.stride(1),
            block_size=block_size,
            D=head_dim,
            D_BLOCK=D_BLOCK,
            BLOCK_N=BLOCK_N,
        )


# ---------------------------------------------------------------------------
#  Triton kernel: update_block_summary_decode (graph-safe, write-on-fill)
#
#  Decode-only summary maintenance with a *fixed* grid (bsz, H) and NO host-side
#  sync — the prerequisite for CUDA-graph capture.
#
#  Key observation (why we do NOT roll a mean every step): a block's summary is
#  only ever *read* by routing once the block is a COMPLETE previous block.
#  `block_inds_to_kv_inds_for_decoding` scores/selects only blocks
#  `[0, current_block)`; the current (partial) block is appended by position and
#  never scored.  So a partial block's summary is never consumed — updating it on
#  every decode step is wasted work.  Instead we write a block's summary exactly
#  ONCE, at the step it fills (`seq_len % block_size == 0`), by reading all
#  `block_size` of its keys straight from the paged `k_buffer` (via req_to_token)
#  and taking their mean.  Non-fill steps (the common case, ~block_size-1 of every
#  block_size steps) do nothing.
#
#  This is exactly what the shared complete-block builder does
#  (`_build_block_summaries_from_cache_kernel`): one reduction over the block's
#  cached keys.  All three build paths (prefix recompute, extend, decode
#  write-on-fill) read the same `k_buffer`, so the decode summary matches the
#  extend / prefix summary bit-for-bit (single fp32 reduction, no incremental
#  accumulation).  A block straddling the prefill->decode boundary is handled for
#  free — at fill time the whole block is read from the cache regardless of which
#  tokens came from prefill vs decode.
#
#  (Decode keeps its own kernel rather than reusing the shared builder because it
#  has a different gating predicate — fill-this-step vs a precomputed block range
#  — and must be a fixed `(bsz, H)` grid with no host-side block-range tensors to
#  stay CUDA-graph capturable.)  Graph-safety: fixed grid, `seq_len % block_size
#  == 0` and slot guards are data-dependent branches *inside* the kernel (capture
#  records only the fixed launch).
# ---------------------------------------------------------------------------


@triton.jit
def _update_block_summary_decode_kernel(
    # -- KV cache key buffer (for one layer) --
    k_buffer_ptr,  # [pool_size, H, D]
    k_buffer_stride_tok,
    k_buffer_stride_head,
    # -- req_to_token: cached token locations --
    req_to_token_ptr,  # [req_pool_size, max_ctx_len]  int32
    req_to_token_stride,
    # -- per-request info --
    seq_lens_ptr,  # [bsz]  int32/int64 – seq len INCLUDING the new token
    req_pool_indices_ptr,  # [bsz]  int64 – row index into req_to_summary / req_to_token
    # -- block mapping --
    req_to_summary_ptr,  # [req_pool_size, max_num_summary]  int32
    req_to_summary_stride,  # stride of dim-0 (= max_num_summary)
    # -- summary buffer (per-layer, already selected before launch) --
    summary_buf_ptr,  # [summary_pool_size+1, H, D]
    summary_buf_stride_slot,
    summary_buf_stride_head,
    # -- constants --
    block_size: tl.constexpr,
    D: tl.constexpr,
    D_BLOCK: tl.constexpr,  # next_power_of_2(D), for masking
    BLOCK_N: tl.constexpr,  # next_power_of_2(block_size), token-axis tile
):
    pid_req = tl.program_id(0)
    pid_head = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_req).to(tl.int32)
    # Act only when a block just FILLED this step (seq_len % block_size == 0).
    # This also cleanly excludes CUDA-graph padding rows: under graph replay a
    # real batch is padded up to the captured bs with seq_len == fill_value (1)
    # and req_pool_idx == 0, and 1 is never a multiple of block_size (> 1), so
    # padding rows take the early return and never touch slot 0's summary.
    # (A data-dependent branch inside the kernel is CUDA-graph safe.)
    if seq_len % block_size != 0:
        return

    req_pool_idx = tl.load(req_pool_indices_ptr + pid_req).to(tl.int64)

    # The block that just completed: tokens [block_start, block_start+block_size).
    block_id = (seq_len - 1) // block_size
    block_start = block_id * block_size

    summary_loc = tl.load(
        req_to_summary_ptr + req_pool_idx * req_to_summary_stride + block_id
    ).to(tl.int32)
    # Sentinel: slot 0 is the padding/unallocated slot.  A completed block always
    # has a slot allocated; guard defensively.
    if summary_loc <= 0:
        return

    d_offsets = tl.arange(0, D_BLOCK)
    d_mask = d_offsets < D

    # Mean of the block's keys, read from the paged cache via req_to_token —
    # identical to the shared complete-block builder
    # (_build_block_summaries_from_cache_kernel): load all block_size token
    # locations at once, do a single 2D gather [BLOCK_N, D_BLOCK], and reduce over
    # the token axis (mirrors seer_attn's _prefill_pool_kernel).
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < block_size  # BLOCK_N == next_pow2(block_size); mask the tail
    tok_locs = tl.load(
        req_to_token_ptr + req_pool_idx * req_to_token_stride + block_start + offs_n,
        mask=n_mask,
        other=0,
    ).to(tl.int64)
    # 2D gather: rows = the block's tokens (scattered in the pool), cols = head_dim.
    k = tl.load(
        k_buffer_ptr
        + tok_locs[:, None] * k_buffer_stride_tok
        + pid_head * k_buffer_stride_head
        + d_offsets[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(k, axis=0) / block_size

    s_ptr = (
        summary_buf_ptr
        + summary_loc * summary_buf_stride_slot
        + pid_head * summary_buf_stride_head
        + d_offsets
    )
    tl.store(s_ptr, mean.to(summary_buf_ptr.dtype.element_ty), mask=d_mask)


def update_block_summary_decode(
    layer: "RadixAttention",
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    forward_batch: "ForwardBatch",
    req_to_token: torch.Tensor,
    req_to_summary: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Graph-safe decode-step summary maintenance (write-on-fill).

    Fixed grid ``(bsz, num_kv_heads)``, no host-side ``.item()`` sync, so it is
    safe to capture in a CUDA graph.  A request writes its block summary only on
    the step the block fills (``seq_len % block_size == 0``); the summary is the
    mean of the block's ``block_size`` keys read from the paged ``k_buffer`` via
    ``req_to_token`` — the same computation the prefill build does
    (:func:`recompute_all_prefix_summaries`), so decode and prefill summaries
    agree exactly.  Non-fill steps and padding rows are skipped in-kernel.

    A partial block's summary is never read by routing (only complete previous
    blocks are scored — see
    :func:`~sglang.srt.layers.attention.blocksparse.common_index_kernels.block_inds_to_kv_inds_for_decoding`),
    so there is no need to roll a mean every step.

    The new decode key must already be written to ``k_buffer`` before this call
    (the base ``forward_decode`` does ``set_kv_buffer`` first), so a block that
    fills this step has all ``block_size`` keys present in the cache.
    """
    token_to_kv_pool: "BlockSparseTokenToKVPool" = forward_batch.token_to_kv_pool
    H = num_kv_heads
    D = head_dim
    bsz = seq_lens.size(0)

    k_buffer = token_to_kv_pool.get_key_buffer(layer.layer_id)
    summary_buf = token_to_kv_pool.get_summary_buffer(layer.layer_id)
    D_BLOCK = triton.next_power_of_2(D)
    BLOCK_N = triton.next_power_of_2(block_size)

    _update_block_summary_decode_kernel[(bsz, H)](
        k_buffer,
        k_buffer.stride(0),
        k_buffer.stride(1),
        req_to_token,
        req_to_token.stride(0),
        seq_lens,
        req_pool_indices,
        req_to_summary,
        req_to_summary.stride(0),
        summary_buf,
        summary_buf.stride(0),
        summary_buf.stride(1),
        block_size=block_size,
        D=D,
        D_BLOCK=D_BLOCK,
        BLOCK_N=BLOCK_N,
    )
