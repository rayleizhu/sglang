from __future__ import annotations

"""Summary-slot bookkeeping for the block-sparse attention backends.

The block-sparse backends (``seer_attn``, ``moba``) keep, alongside the KV
cache, a pool of per-block *summary* vectors plus a ``req_to_summary`` table
mapping each request's blocks to slots in that pool.  Those slots have to be
allocated as new tokens are written and freed when a request is released --
i.e. at exactly the points where KV slots are allocated and freed.

This module holds that bookkeeping so ``mem_cache/common.py`` (the generic
allocation path shared by every backend) keeps only thin call sites.  All three
helpers are no-ops unless the pools in play are the block-sparse ones, so they
are safe to call unconditionally.
"""

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.allocator import BlockSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BlockSparseReqToTokenPool, ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def _maybe_alloc_summary_slots(
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator,
    prefix_lens: torch.Tensor,  # [bsz] start of new tokens (0-indexed), GPU
    seq_lens_after: torch.Tensor,  # [bsz] end of new tokens (exclusive), GPU
    req_pool_indices: torch.Tensor,  # [bsz], GPU
    prefix_lens_cpu: torch.Tensor,  # [bsz] CPU mirror of prefix_lens
    seq_lens_after_cpu: torch.Tensor,  # [bsz] CPU mirror of seq_lens_after
    is_first_extend_cpu: Optional[
        torch.Tensor
    ] = None,  # [bsz] bool, True if first extend (new req / cache hit / 1st chunk)
):
    """Allocate summary pool slots for blocks that need them.

    ``is_first_extend_cpu`` distinguishes whether each request is seeing its
    first extend (True) or is a continuation chunk in chunked prefill (False).

    * **First extend** (new request, prefix cache hit, 1st chunk of chunked
      prefill, or session 2nd turn): allocate summary slots starting from
      block 0 so that prefix blocks also get slots.
    * **Continuation chunk** (2nd+ chunk of chunked prefill): prefix blocks
      already have slots from the previous chunk, so start from
      ``ceil(prefix / block_size)``.
    * **Decode** (``is_first_extend_cpu is None``): all prefix blocks already
      have slots; only allocate for newly touched blocks.

    Uses a Triton kernel to write slots into ``req_to_summary`` — no Python
    loop or ``.item()`` device sync.  ``total_new`` is computed from CPU
    mirrors so that ``.sum().item()`` does not trigger a GPU-CPU sync.
    """
    if not isinstance(token_to_kv_pool_allocator, BlockSparseTokenToKVPoolAllocator):
        return
    if not isinstance(req_to_token_pool, BlockSparseReqToTokenPool):
        return

    block_size = req_to_token_pool.block_size
    bsz = prefix_lens.size(0)

    # --- CPU path: compute alloc_start_block per request ---
    if is_first_extend_cpu is not None:
        # Extend path:
        #   first extend  → start from block 0 (prefix blocks need slots too)
        #   continuation  → start after already-allocated prefix blocks
        alloc_start_block_cpu = torch.where(
            is_first_extend_cpu,
            torch.zeros_like(prefix_lens_cpu),
            (prefix_lens_cpu + block_size - 1) // block_size,
        )
    else:
        # Decode path: all prefix blocks already have slots.
        alloc_start_block_cpu = (prefix_lens_cpu + block_size - 1) // block_size

    last_block_cpu = (seq_lens_after_cpu - 1) // block_size
    num_new_per_req_cpu = last_block_cpu - alloc_start_block_cpu + 1
    total_new = num_new_per_req_cpu.sum().item()  # CPU .item(), no GPU sync
    if os.environ.get("SGLANG_SEER_DEBUG_SLOTS") == "1":
        _ife = None if is_first_extend_cpu is None else is_first_extend_cpu.tolist()
        print(
            f"[seer-slots] alloc call: first_extend={_ife} "
            f"prefix={prefix_lens_cpu.tolist()} after={seq_lens_after_cpu.tolist()} "
            f"start_blk={alloc_start_block_cpu.tolist()} "
            f"last_blk={last_block_cpu.tolist()} num_new={num_new_per_req_cpu.tolist()} "
            f"total_new={total_new} avail={token_to_kv_pool_allocator.summary_available_size()}",
            flush=True,
        )
    if total_new == 0:
        return

    summary_locs = token_to_kv_pool_allocator.alloc_summary(total_new)
    if summary_locs is None:
        # Hard failure rather than a silent warning.  If we return here the
        # affected blocks keep slot 0 (the padding sentinel) in req_to_summary,
        # so the gate scores them as empty blocks and silently produces WRONG
        # output instead of crashing.  The summary pool is sized as
        # ceil(kv_size / block_size) + max_num_reqs, which covers the full KV
        # cache *plus* the per-request partial-tail-block over-allocation
        # (Σ ceil(len_i/block) ≤ kv_size/block + num_reqs).  KV-cache admission
        # should therefore exhaust before the summary pool can; reaching this
        # point means a slot leak or an accounting bug in the free path, which
        # must surface, not degrade accuracy.
        raise RuntimeError(
            "Block-sparse summary pool exhausted: requested "
            f"{total_new} slots but only "
            f"{token_to_kv_pool_allocator.summary_available_size()} free. "
            "The summary pool is sized to cover the full KV cache, so this "
            "indicates a summary-slot leak in the release path rather than "
            "legitimate memory pressure."
        )

    # Build cumsum offsets from CPU-side num_new_per_req, then transfer to GPU.
    cum_new_cpu = torch.zeros(bsz + 1, dtype=torch.int32)
    cum_new_cpu[1:] = torch.cumsum(num_new_per_req_cpu.to(torch.int32), dim=0)
    cum_new = cum_new_cpu.to(prefix_lens.device, non_blocking=True)

    # Transfer alloc_start_block to GPU for the Triton kernel.
    alloc_start_block = alloc_start_block_cpu.to(
        prefix_lens.device, non_blocking=True
    ).to(torch.int32)

    _write_summary_slots_kernel[(bsz,)](
        req_to_token_pool.req_to_summary,
        req_to_token_pool.req_to_summary.stride(0),
        req_pool_indices,
        alloc_start_block,
        cum_new,
        summary_locs.to(torch.int32),
    )


@triton.jit
def _write_summary_slots_kernel(
    req_to_summary_ptr,  # [req_pool_size, max_num_summary]
    req_to_summary_stride,  # stride of dim-0
    req_pool_indices_ptr,  # [bsz]
    alloc_start_blocks_ptr,  # [bsz] int32, first block to allocate for each request
    cum_new_ptr,  # [bsz+1] int32, cumsum offsets into summary_locs
    summary_locs_ptr,  # [total_new] int32
):
    """Write allocated summary slots into req_to_summary.  One program per request.

    req_to_summary[req_pool_indices[pid]][start_blk+i] = summary_locs[cum_new[pid]+i]
        for i in range(num_new_per_req[pid])
    """
    pid = tl.program_id(0)
    req_idx = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    start_blk = tl.load(alloc_start_blocks_ptr + pid).to(tl.int32)

    # n = number of blocks to allocate for this request, encoded in cum_new
    offset = tl.load(cum_new_ptr + pid).to(tl.int32)
    next_offset = tl.load(cum_new_ptr + pid + 1).to(tl.int32)
    n = next_offset - offset

    for i in range(n):
        slot = tl.load(summary_locs_ptr + offset + i)
        tl.store(
            req_to_summary_ptr + req_idx * req_to_summary_stride + start_blk + i,
            slot,
        )


def _maybe_free_summary_for_release(req: Req, tree_cache: BasePrefixCache):
    """Free summary pool slots when a request is released."""
    allocator = tree_cache.token_to_kv_pool_allocator
    if not isinstance(allocator, BlockSparseTokenToKVPoolAllocator):
        return

    req_to_token_pool = tree_cache.req_to_token_pool
    if not isinstance(req_to_token_pool, BlockSparseReqToTokenPool):
        return

    # NOTE: currently, there are several defensive guards in this function,
    # such as req.kv_committed_len <= 0, valid_mask.any(), summary_row[valid_mask] = 0,
    # they can be removed for simplicity, but we keep them for now for safety.

    # Precisely compute how many blocks this request occupies, then free
    # only those slots — avoids scanning/zeroing the entire row.
    seq_len = req.kv_committed_len
    if seq_len <= 0: # defensive guard
        return
    block_size = req_to_token_pool.block_size
    num_blocks = (seq_len + block_size - 1) // block_size
    summary_row = req_to_token_pool.req_to_summary[req.req_pool_idx, :num_blocks]
    valid_mask = summary_row > 0
    if valid_mask.any():
        allocator.free_summary(summary_row[valid_mask].long())
        summary_row[valid_mask] = 0  # precise zero to prevent stale slot reuse
