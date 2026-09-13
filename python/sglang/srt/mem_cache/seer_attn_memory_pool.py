# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Memory pools for the SeerAttention-R decode-sparse attention backend.

SeerAttention-R needs two auxiliary caches on top of the regular paged KV
cache (k_buffer / v_buffer):

* **summary cache** — one *gated compressed-K* vector per completed logical
  KV block, of width ``gate_hidden_size``.  Used only by the AttnGate to score
  blocks; the actual attention reads the real KV of the selected blocks.  This
  reuses SGLang's existing block-sparse summary machinery
  (:class:`BlockSparseTokenToKVPool` ``summary_buffer`` +
  :class:`BlockSparseReqToTokenPool` ``req_to_summary`` +
  :class:`BlockSparseTokenToKVPoolAllocator` slot free-list), with
  ``summary_head_dim`` set to ``gate_hidden_size``.

* **rolling accumulator** — a tiny per-request running ``[max | min | sum]``
  reduction of the *raw* (pre-RoPE, post-qknorm) keys of the **current,
  not-yet-complete** block.  The gate compresses a block from the
  ``max|min|avg`` pool of its raw keys, but SGLang's main KV cache stores
  *post-RoPE* keys — so we fold each new key into this accumulator instead of
  buffering the whole block.  A block that completes during decode always holds
  exactly ``block_size`` tokens, so ``avg = sum / block_size`` recovers the
  average at fill time.  Shape per layer: ``[max_num_reqs, kv_heads,
  3 * head_dim]`` fp32 (the three concatenated reductions); this is ~``block_size``×
  smaller than buffering the raw block, and ``max``/``min`` become
  reduction-order-independent (bit-exact) while ``sum`` stays fp32.

By subclassing the existing block-sparse pools, the summary-slot allocator and
``_maybe_alloc_summary_slots`` (which dispatch on ``isinstance``) work for the
SeerAttention backend without modification.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import (
    BlockSparseReqToTokenPool,
    BlockSparseTokenToKVPool,
)


class SeerAttnReqToTokenPool(BlockSparseReqToTokenPool):
    """Request->token pool with the ``req_to_summary`` block-index table.

    Identical to :class:`BlockSparseReqToTokenPool`; defined as a distinct type
    so the backend can be selected/branched on cleanly.
    """


class SeerAttnTokenToKVPool(BlockSparseTokenToKVPool):
    """KV pool with a gated-compressed-K summary cache + a raw-K rolling cache.

    ``summary_head_dim`` is the AttnGate hidden size (the width of a compressed
    block vector).  ``rolling_buffer[layer]`` holds a running ``[max | min |
    sum]`` reduction (fp32, width ``3 * head_dim``) of the raw keys of the
    current incomplete block for every request — not the raw keys themselves.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        block_size: int,
        gate_hidden_size: int,
        max_num_reqs: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.gate_hidden_size = gate_hidden_size
        # routing_mode is unused by SeerAttention (gate decides routing) but the
        # parent stores it; pass a benign value.
        super().__init__(
            size=size,
            page_size=page_size,
            block_size=block_size,
            routing_mode="shared",
            summary_head_dim=gate_hidden_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            max_num_reqs=max_num_reqs,
        )

    def _create_buffers(self):
        super()._create_buffers()  # k_buffer, v_buffer, summary_buffer
        # Per-layer rolling accumulator: a running [max | min | sum] reduction
        # (fp32, width 3 * head_dim) of the raw (pre-RoPE, post-qknorm) keys of
        # the current incomplete block of each request.  Self-resetting: the
        # fold kernel overwrites the accumulator on the first token of a new
        # block (pos_in_block == 0), so no identity pre-fill is required.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.rolling_buffer = [
                    torch.zeros(
                        (
                            self.max_num_reqs,
                            self.head_num,
                            3 * self.head_dim,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_rolling_buffer(self, layer_id: int) -> torch.Tensor:
        return self.rolling_buffer[layer_id - self.start_layer]
