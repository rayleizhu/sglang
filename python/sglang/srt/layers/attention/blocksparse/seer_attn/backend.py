# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""SeerAttention-R decode-sparse attention backend for SGLang.

Implements the inference path of SeerAttention-R
(https://github.com/microsoft/SeerAttention, arXiv:2506.08889) on top of the
shared block-sparse machinery in
:class:`~sglang.srt.layers.attention.blocksparse.base_backend.BlockSparseAttnBackend`:

* **Prefill / extend** is *dense* (delegated to the shared flashinfer paged
  prefill path).  During extend we also build the per-block *gated compressed-K*
  summaries for every complete block (used by the gate at decode time), and seed
  the per-request *rolling accumulator* (``[max | min | sum]`` of the raw keys
  of the final partial block).

* **Decode** is *block-sparse*.  For each step and layer ``>= start_layer``:

  1. fold the new token's raw (pre-RoPE, post-k_norm) key into the rolling
     ``[max | min | sum]`` accumulator; if the current block just *filled up*,
     gate-compress it (``avg = sum / block_size``) into the summary cache
     (RoPE'd at the completing-token position).
  2. run the AttnGate: project Q (RoPE'd at the current position), score it
     against the per-block compressed-K summaries, and select the
     top-``block_budget`` *previous* blocks by score (shared selector; the
     monotonic softmax of the reference is unnecessary for a top-k and is
     dropped).
  3. run the shared flashinfer virtual-batch block-sparse decode over the
     selected blocks' real KV (the current partial block is always attended,
     appended by position by the shared kv-index expansion).

  Layers ``< start_layer`` run dense decode (shared flashinfer GQA path).

This backend differs from MoBA only in the block *representation* (learned
gated compressed-K + rolling accumulator vs. a rolling mean) and the block
*selection* (gate top-k vs. dot-product top-k); both reuse the same
flashinfer dense/sparse plumbing.

The gate math (:mod:`sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate`)
is a bit-exact port of the reference; RoPE and the rolling/summary maintenance
are done against SGLang's paged KV layout.

The model passes the gate's required tensors through the ``RadixAttention``
``**kwargs`` channel (see ``models/qwen3_seer.py``):

* ``seer_q_nope``: ``[num_tokens, num_q_heads, head_dim]`` post-q_norm, pre-RoPE
  query.
* ``seer_k_nope``: ``[num_tokens, num_kv_heads, head_dim]`` post-k_norm, pre-RoPE
  key.
* ``seer_gate``: the per-layer :class:`AttnGate` module (holds trained weights).
* ``seer_rope_inv_freq``: ``[gate_hidden_size//2]`` fp32 NeoX inverse
  frequencies; the gate RoPE is computed in-kernel from these.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.blocksparse.base_backend import (
    BlockSparseAttnBackend,
)
from sglang.srt.layers.attention.blocksparse.common_index_kernels import (
    compute_active_block_ids,
)
from sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate import (
    rope_gate_query,
)
from sglang.srt.layers.attention.blocksparse.seer_attn.cache_kernels import (
    build_prefill_summary_schedule,
    update_summary_cache_decode,
    update_summary_cache_prefill,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class SeerAttnBackend(BlockSparseAttnBackend):
    """Block-sparse decode attention backend (SeerAttention-R)."""

    def _init_sparse_config(self, model_runner: "ModelRunner") -> None:
        cfg = model_runner.model_config.seer_attn
        self.block_size = cfg.block_size
        self.gate_hidden_size = cfg.gate_hidden_size
        # SeerAttention-R only supports the token-budget sparsity mode here.
        self.token_budget = cfg.token_budget
        self.block_budget = cfg.token_budget // cfg.block_size
        # The shared decode plan uses a fixed budget of `topk` previous blocks.
        self.topk = self.block_budget
        self.start_layer = cfg.start_layer

        # Block summary = gated compressed-K of width gate_hidden_size.
        self.summary_head_dim = self.gate_hidden_size

        # Guard the assumptions this backend silently relies on so a
        # misconfiguration fails loudly here instead of producing wrong results
        # or a cryptic deep-stack error later.
        #
        # 1. page_size must be 1: the shared kv-index expansion gathers paged KV
        #    per *token* via req_to_token (one slot per logical token), not per
        #    page, so a page_size > 1 layout would mis-index the cache.
        if model_runner.page_size != 1:
            raise NotImplementedError(
                "seer_attn attention backend only supports page_size == 1 "
                f"(got page_size={model_runner.page_size}). The block-sparse "
                "decode path gathers paged KV per token via req_to_token."
            )
        # 2. The request pool must carry the gate's summary-slot table.
        if not hasattr(model_runner.req_to_token_pool, "req_to_summary"):
            raise RuntimeError(
                "seer_attn attention backend requires a request pool with a "
                "'req_to_summary' table (SeerAttnReqToTokenPool), got "
                f"{type(model_runner.req_to_token_pool).__name__}. This usually "
                "means the KV-cache pool selection did not route to the "
                "SeerAttention pools."
            )
        # 3. Disaggregated (PD) serving is unsupported.  The summary cache and
        #    the per-request rolling accumulator are built *only* by the local
        #    dense prefill (_build_block_summary_extend).  On a PD decode node
        #    the KV is transferred from the prefill node and no local prefill
        #    runs, so neither auxiliary cache is ever populated -> the gate
        #    would score against zero/garbage summaries.  Fail loudly until the
        #    transfer path also carries (or rebuilds) these caches.
        sa = model_runner.server_args
        if getattr(sa, "disaggregation_mode", "null") != "null":
            raise NotImplementedError(
                "seer_attn attention backend does not support disaggregated "
                f"(PD) serving (disaggregation_mode={sa.disaggregation_mode!r}). "
                "The gate summary cache and rolling accumulator are built only "
                "by the local dense prefill; a decode node that receives KV "
                "without running prefill would have neither populated."
            )
        # 4. Mixed prefill+decode batches are unsupported.  init_forward_metadata
        #    treats every non-decode forward as a pure extend (single-chunk from
        #    position 0), and forward_extend/forward_decode are dispatched per
        #    ForwardMode -- a MIXED batch would run the dense extend path for
        #    decode rows, skipping the rolling fold + sparse decode entirely.
        #    Disabling radix + chunked prefill already prevents MIXED on CUDA;
        #    this guard makes the dependency explicit (enable_mixed_chunk only
        #    takes effect together with chunked prefill, but assert directly on
        #    it so a future change to that coupling fails here, not silently).
        if getattr(sa, "enable_mixed_chunk", False):
            raise NotImplementedError(
                "seer_attn attention backend does not support "
                "enable_mixed_chunk: a mixed prefill+decode batch would run the "
                "dense extend path for decode rows and skip the rolling-cache "
                "fold and block-sparse decode."
            )
        # 5. Streaming sessions are unsupported.  SessionAwareCache transfers KV
        #    ownership from a finishing request to a SessionSlot by setting
        #    req.req_pool_idx = None, which makes release_kv_cache take the
        #    early-return transfer path *before* _maybe_free_summary_for_release
        #    runs -> the per-block summary slots leak (and the rolling
        #    accumulator's "fresh prefill from pos 0" invariant no longer holds
        #    across turns).  release_kv_cache only asserts against this, which
        #    is stripped under `python -O`, so guard it here at startup instead
        #    of failing on the first finished streaming request.
        if getattr(sa, "enable_streaming_session", False):
            raise NotImplementedError(
                "seer_attn attention backend does not support "
                "enable_streaming_session: SessionAwareCache nulls req_pool_idx "
                "to transfer KV ownership, which would leak the per-block gate "
                "summary slots (they are freed keyed on req_pool_idx) and break "
                "the rolling accumulator's fresh-prefill-per-turn invariant."
            )
        # 6. Speculative decoding is unsupported.  alloc_for_decode (and the
        #    summary-slot allocation it drives) assumes exactly one new token per
        #    request per step; a draft/verify step decodes multiple tokens at
        #    once, which the rolling fold + per-block summary write path does not
        #    handle.  Without this guard the failure is a deep-stack RuntimeError
        #    from alloc_for_decode mid-serving; surface it cleanly at startup.
        if getattr(sa, "speculative_algorithm", None) is not None:
            raise NotImplementedError(
                "seer_attn attention backend does not support speculative "
                f"decoding (speculative_algorithm={sa.speculative_algorithm!r}). "
                "The rolling-cache fold and per-block summary write assume one "
                "new token per request per decode step."
            )

        # Per-forward, layer-independent prefill summary/rolling schedule, built
        # once per extend forward in init_forward_metadata and reused by every
        # seer layer's _build_block_summary_extend.  Decode never uses it.
        self._summary_schedule = None

    # ------------------------------------------------------------------
    # Sparse-layer gating
    # ------------------------------------------------------------------
    def _is_sparse_layer(self, layer: "RadixAttention", **kwargs) -> bool:
        # SeerAttention-R additionally requires the gate tensors for this layer.
        return layer.layer_id >= self.start_layer and "seer_gate" in kwargs

    # ------------------------------------------------------------------
    # Block representation (gated compressed-K + rolling accumulator)
    # ------------------------------------------------------------------
    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        super().init_forward_metadata(forward_batch)
        # Build the layer-independent prefill summary/rolling schedule once per
        # extend forward (it depends only on extend_seq_lens / req_pool_indices /
        # req_to_summary, identical across layers), so every seer layer's
        # _build_block_summary_extend reuses it instead of rebuilding it (with a
        # host sync) per layer.  init_forward_metadata is called exactly once
        # before each forward, so this is the right place.  Decode never uses it.
        #
        # CHUNKED PREFILL: the schedule takes extend_prefix_lens (tokens already
        # summarized by earlier chunks of the same request) so a chunk starting
        # mid-request writes its blocks to the correct request-global slots / RoPE
        # positions.  Requires each chunk boundary block_size-aligned (asserted in
        # build_prefill_summary_schedule).  For a fresh single-chunk prefill from 0,
        # extend_prefix_lens is all-zeros and this reduces to the old behavior.
        #
        # MIXED (prefill+decode interleaved) is still avoided (forward_mixed is
        # NPU-only), so is_extend() here only ever sees a pure extend.
        self._summary_schedule = None
        if forward_batch.forward_mode.is_extend():
            self._summary_schedule = build_prefill_summary_schedule(
                forward_batch.extend_seq_lens,
                forward_batch.req_pool_indices,
                self.block_size,
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                extend_prefix_lens=forward_batch.extend_prefix_lens,
                extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            )

    def _build_block_summary_extend(
        self, layer: "RadixAttention", k, forward_batch: "ForwardBatch", **kwargs
    ) -> None:
        """Build gate summaries for all complete blocks during prefill.

        Prefill is dense, but the gate needs per-block compressed-K summaries for
        the subsequent decode steps.  ``update_summary_cache_prefill`` seeds both
        prefill caches (per-complete-block gated summary + per-request rolling
        accumulator of the trailing partial block) with two Triton kernels
        sharing one host schedule (built once per forward in
        ``init_forward_metadata`` -> ``self._summary_schedule``).

        Supports chunked prefill: the schedule carries per-request
        ``prefix_blocks`` (from ``extend_prefix_lens``) so a chunk starting
        mid-request writes to the correct request-global summary slots / RoPE
        positions, provided each chunk boundary is block_size-aligned.
        """
        gate = kwargs["seer_gate"]
        k_nope = kwargs["seer_k_nope"]  # [total_tokens, Hk, D]
        inv_freq = kwargs["seer_rope_inv_freq"]  # [gate_dim//2] fp32

        pool = forward_batch.token_to_kv_pool
        rolling = pool.get_rolling_buffer(layer.layer_id)
        summary_buf = pool.get_summary_buffer(layer.layer_id)

        schedule = self._summary_schedule
        assert schedule is not None, (
            "seer_attn: _summary_schedule was not built in init_forward_metadata "
            "before _build_block_summary_extend; the forward call contract is broken."
        )

        update_summary_cache_prefill(
            schedule,
            k_nope,
            summary_buf,
            rolling,
            self._model_runner.req_to_token_pool.req_to_summary,
            gate.attngate_linear_k.weight,
            gate.attngate_knorm.weight,
            inv_freq,
            block_size=self.block_size,
            eps=gate.attngate_knorm.variance_epsilon,
        )

    def _update_block_summary_decode(
        self, layer: "RadixAttention", k, forward_batch: "ForwardBatch", **kwargs
    ) -> None:
        """Rolling-accumulator fold + (on block fill) summary compression.

        Folds the new raw K into the rolling ``[max | min | sum]`` accumulator;
        when ``seq_len % block_size == 0`` (the block just filled), recovers the
        block's max|min|avg, projects with the gate K-branch, norms, RoPE at the
        *current* token position, and writes it into the summary slot.  Fused
        into one graph-safe Triton kernel.
        """
        gate = kwargs["seer_gate"]
        k_nope = kwargs["seer_k_nope"]  # [bsz, Hk, D] post-k_norm, pre-RoPE
        inv_freq = kwargs["seer_rope_inv_freq"]

        pool = forward_batch.token_to_kv_pool
        rolling = pool.get_rolling_buffer(layer.layer_id)  # [R, Hk, 3*D] fp32
        summary_buf = pool.get_summary_buffer(layer.layer_id)
        req_to_summary = self._model_runner.req_to_token_pool.req_to_summary

        seq_lens = forward_batch.seq_lens
        bsz = seq_lens.shape[0]
        k_dec = k_nope.contiguous().view(bsz, self.num_kv_heads, self.head_dim)
        update_summary_cache_decode(
            k_dec,
            rolling,
            seq_lens.to(torch.int32),
            forward_batch.req_pool_indices,
            req_to_summary,
            summary_buf,
            gate.attngate_linear_k.weight,
            gate.attngate_knorm.weight,
            inv_freq,
            block_size=self.block_size,
            norm_eps=gate.attngate_knorm.variance_epsilon,
        )

    # ------------------------------------------------------------------
    # Block selection (gate scoring -> top-block_budget)
    # ------------------------------------------------------------------
    def _select_active_blocks(
        self, layer: "RadixAttention", q, forward_batch: "ForwardBatch", **kwargs
    ) -> torch.Tensor:
        """Run the gate scoring and return active *previous* block ids.

        Returns ``[bsz, num_kv_heads, block_budget]`` int32, ``-1`` padded.

        SeerAttention-R's gate query is the model query projected to gate space
        (+ qnorm) and then RoPE'd at the *current* token position.  That
        projection and RoPE are the gate's *representation* of the query; once
        applied, scoring it against the (also-RoPE'd) per-block compressed-K
        summaries is a plain dot product, so block selection reuses the shared
        :func:`compute_active_block_ids`.  No forced bands (``num_init`` /
        ``num_local`` = 0): the gate alone decides the top-``block_budget``
        previous blocks; the current/partial block is appended downstream by
        position.

        Both the Q projection (+ qnorm) and the NeoX RoPE here are static-shape
        torch ops (``block_budget`` and the RoPE width are fixed for a given
        model/config), so the whole selection stays CUDA-graph capturable.
        """
        gate = kwargs["seer_gate"]
        q_nope = kwargs["seer_q_nope"]  # [T, Hq, D] post-q_norm, pre-RoPE
        inv_freq = kwargs["seer_rope_inv_freq"]  # [gate_dim//2] fp32

        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices
        bsz = seq_lens.shape[0]

        q_dec = q_nope.contiguous().view(bsz, self.num_qo_heads, self.head_dim)
        # Q branch: project (+ qnorm) -> [bsz, Hk, gate_dim].
        q_gate = gate.project_q(q_dec)  # [bsz, Hk, gate_dim]

        # Apply NeoX RoPE at each request's current token position (seq_len - 1),
        # matching the position the summary K vectors were RoPE'd at.  Fused into
        # a single Triton kernel (grid (bsz, Hk), fp32 rotation to match the
        # K-side summary numerics) instead of the multi-op torch path — fewer
        # launches, no cos/sin/emb intermediates.  Hoisting RoPE here (out of the
        # score kernel) is what lets the dot-product scoring reuse the shared
        # selector; the kernel keeps static shapes, so it stays graph-safe.
        q_gate = rope_gate_query(q_gate, seq_lens, inv_freq)

        req_to_summary = self._model_runner.req_to_token_pool.req_to_summary
        summary_buf = forward_batch.token_to_kv_pool.get_summary_buffer(
            layer.layer_id
        )  # [pool+1, Hk, gate_dim]

        return compute_active_block_ids(
            q_gate,
            summary_buf,
            seq_lens,
            req_pool_indices,
            req_to_summary,
            block_size=self.block_size,
            topk=self.block_budget,
            num_init_blocks=0,
            num_local_blocks=0,
        )
