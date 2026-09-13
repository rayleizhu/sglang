from __future__ import annotations

"""MoBA block-sparse attention backend.

Block representation is a simple rolling *mean* of the keys of each logical KV
block; block selection is per-(request, kv-head) top-k by query-summary dot
product, with forced leading "sink" and trailing "local" bands.  Prefill is
dense; decode is block-sparse via flashinfer virtual batching.

All the flashinfer plumbing (dense prefill, virtual-batch sparse decode, the
kv-index expansion) lives in
:class:`~sglang.srt.layers.attention.blocksparse.base_backend.BlockSparseAttnBackend`;
this subclass only supplies the rolling-mean summary maintenance and the top-k
routing.

This backend mirrors SeerAttention-R
(:mod:`sglang.srt.layers.attention.blocksparse.seer_attn.backend`); both reuse
the same dense/sparse flashinfer plumbing and the shared graph-safe block
selector (:func:`compute_active_block_ids`).  They differ only in the block
*representation* (rolling mean of the cached keys vs. a learned gated
compressed-K) and the block *selection* (dot-product top-k vs. the gate's).

CUDA graph: the decode path is graph-capturable.  The shared virtual-batch
decode (base backend) expands KV indices into a fixed pre-allocated buffer, and
the per-step summary maintenance uses a fixed-grid, sync-free decode kernel
(:func:`~sglang.srt.layers.attention.blocksparse.moba.cache_kernels.update_block_summary_decode`)
that writes a block's summary only on the step it fills (a partial block's
summary is never scored, so there is no need to roll a mean every step).
Extend is never graph-captured (SGLang only captures ``ForwardMode.DECODE``), so
the extend-time summary build keeps its host-scheduled ``.item()`` grid sizing.
Unlike SeerAttention-R, MoBA supports radix/prefix cache and chunked prefill
(those touch only the never-captured extend path).
"""

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.blocksparse.base_backend import (
    BlockSparseAttnBackend,
)
from sglang.srt.layers.attention.blocksparse.moba.cache_kernels import (
    recompute_all_prefix_summaries,
    update_block_summary,
    update_block_summary_decode,
)
from sglang.srt.layers.attention.blocksparse.common_index_kernels import (
    compute_active_block_ids,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class MobaAttnBackend(BlockSparseAttnBackend):
    """Block-sparse decode attention backend (MoBA rolling-mean routing)."""

    def _init_sparse_config(self, model_runner: "ModelRunner") -> None:
        cfg = model_runner.model_config.blocksparse_attn
        self.block_size = cfg.block_size
        self.topk = cfg.topk
        # Summary = rolling mean of the block's keys (same width as head_dim).
        self.summary_head_dim = self.head_dim
        # MoBA applies sparse routing on every layer.
        self.start_layer = 0

        # Forced-selection bands for decode routing, shared across kv-heads:
        #   num_init_blocks  : leading sink blocks  [0, num_init)
        #   num_local_blocks : trailing previous blocks before the current one
        #                      (the self/current block is always attended and
        #                      appended separately, so it is excluded here)
        # Defaults (1, 0) keep only the attention-sink block (prior behavior).
        # TODO: make these configurable via model config / env once tuned.
        self.num_init_blocks = 1
        self.num_local_blocks = 0

        # Guard the layout assumption the shared kv-index expansion relies on so
        # a misconfiguration fails loudly here rather than mis-indexing the cache
        # deep in the decode path: page_size must be 1, because the virtual-batch
        # decode gathers paged KV per *token* via req_to_token (one slot per
        # logical token), not per page.  (Same guard as SeerAttnBackend.)
        if model_runner.page_size != 1:
            raise NotImplementedError(
                "moba attention backend only supports page_size == 1 "
                f"(got page_size={model_runner.page_size}). The block-sparse "
                "decode path gathers paged KV per token via req_to_token."
            )

    # ------------------------------------------------------------------
    # Block representation (rolling mean of keys)
    # ------------------------------------------------------------------
    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        super().init_forward_metadata(forward_batch)
        # On extend, recompute the prefix block summaries for *all* layers at
        # once here (instead of per-layer in forward_extend), avoiding N_layers
        # device syncs.  init_forward_metadata is called exactly once before
        # each forward, so this is the right place for layer-independent setup.
        if not forward_batch.forward_mode.is_decode_or_idle():
            recompute_all_prefix_summaries(
                forward_batch,
                self._model_runner.req_to_token_pool.req_to_token,
                self._model_runner.req_to_token_pool.req_to_summary,
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )

    def _build_block_summary_extend(
        self, layer: "RadixAttention", k: torch.Tensor, forward_batch: "ForwardBatch", **kwargs
    ) -> None:
        # Build summaries for the new COMPLETE blocks added by this extend, read
        # from the paged cache (this layer's keys are already in k_buffer — base
        # forward_extend set_kv_buffer runs first).  The trailing partial block is
        # skipped (lazy: built on fill during decode).  Prefix block summaries are
        # built once for all layers in init_forward_metadata.
        update_block_summary(
            layer,
            forward_batch,
            self._model_runner.req_to_token_pool.req_to_token,
            self._model_runner.req_to_token_pool.req_to_summary,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    def _update_block_summary_decode(
        self, layer: "RadixAttention", k: torch.Tensor, forward_batch: "ForwardBatch", **kwargs
    ) -> None:
        # Update the block summary on the step a block fills.  A partial block's
        # summary is never scored by routing (only complete previous blocks are),
        # so rather than roll a mean every step we write the summary exactly once
        # — when seq_len % block_size == 0 — as the mean of the just-completed
        # block's keys read from the paged cache (identical to the prefill build).
        # Graph-safe: fixed (bsz, H) grid, no host `.item()` sync.  The new key is
        # already in k_buffer (base forward_decode set_kv_buffer runs first).
        update_block_summary_decode(
            layer,
            forward_batch.seq_lens,
            forward_batch.req_pool_indices,
            forward_batch,
            self._model_runner.req_to_token_pool.req_to_token,
            self._model_runner.req_to_token_pool.req_to_summary,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    # ------------------------------------------------------------------
    # Block selection (per-group top-k by query-summary similarity)
    # ------------------------------------------------------------------
    def _select_active_blocks(
        self, layer: "RadixAttention", q: torch.Tensor, forward_batch: "ForwardBatch", **kwargs
    ) -> torch.Tensor:
        # q: [bsz, num_q_heads, head_dim].  Intra-group query merging (mean over
        # the GQA group) is an algorithm choice done here, outside the kernel.
        bsz = forward_batch.seq_lens.size(0)
        token_to_kv_pool = forward_batch.token_to_kv_pool
        summary_buffer = token_to_kv_pool.get_summary_buffer(layer.layer_id)
        req_to_summary = self._model_runner.req_to_token_pool.req_to_summary

        q_retrieval = q.view(
            bsz, self.num_kv_heads, self.gqa_group_size, self.head_dim
        ).mean(dim=2)  # [bsz, num_kv_heads, head_dim]

        return compute_active_block_ids(
            q_retrieval,
            summary_buffer,
            forward_batch.seq_lens,
            forward_batch.req_pool_indices,
            req_to_summary,
            block_size=self.block_size,
            topk=self.topk,
            num_init_blocks=self.num_init_blocks,
            num_local_blocks=self.num_local_blocks,
        )
