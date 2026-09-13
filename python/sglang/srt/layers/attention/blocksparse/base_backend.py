from __future__ import annotations

"""Abstract parent backend for block-level sparse attention.

This module factors the *representation-agnostic* machinery shared by every
block-sparse attention backend (MoBA, SeerAttention-R, ...) out of the concrete
backends:

* dense prefill / dense decode via flashinfer paged wrappers,
* block-sparse decode via flashinfer **virtual batching** — the GQA group axis
  is folded into the batch axis so each (request, kv-head) becomes an
  independent length-``G`` "virtual request" that attends only its selected
  blocks' KV (mapped through
  :func:`~sglang.srt.layers.attention.blocksparse.common_index_kernels.block_inds_to_kv_inds_for_decoding`),
* a ``start_layer`` split (layers ``< start_layer`` run dense decode).

The parts that *differ* between backends — how a block is summarised
(representation) and how blocks are selected (routing) — are abstract methods
implemented by subclasses:

* :meth:`_build_block_summary_extend`
* :meth:`_update_block_summary_decode`
* :meth:`_select_active_blocks`

A subclass that needs per-forward, layer-independent prefill state (e.g.
SeerAttention-R's summary/rolling schedule) builds it by overriding
:meth:`init_forward_metadata` (call ``super().init_forward_metadata(...)`` first,
then build and stash it on ``self``).

The decode ``active_block_ids`` contract is uniform across backends:
``[bsz, num_kv_heads, topk]`` int32, ``-1`` padded, ascending, holding only the
selected *previous* blocks.  The current (partial) block is appended
unconditionally by position inside ``block_inds_to_kv_inds_for_decoding`` — so
it is always attended and must NOT be included in ``active_block_ids``.

CUDA graph: the virtual-batch decode is graph-capturable.  The decode-time KV
index expansion writes into a *fixed* pre-allocated buffer
(``self.decode_kv_indices_buf``) every layer — both eager and graph paths share
this single in-place write, so the flashinfer wrapper's ``_paged_kv_indices_buf``
never has to be rebound to a fresh allocation (which is what previously broke
graph capture).  The three graph hooks below mirror the standard flashinfer
backend (capture builds ``use_cuda_graph=True`` wrappers bound to the fixed
buffers and swaps ``begin_forward`` for ``fast_decode_plan``; replay re-plans
in place).  A subclass whose per-step summary update is not itself graph-safe
(e.g. one that needs a ``.item()`` host sync) must keep graph disabled at the
model-runner level; MoBA and SeerAttention-R use fixed-grid, sync-free per-step
kernels, so both are graph-safe.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.blocksparse.common_index_kernels import (
    block_inds_to_kv_inds_for_decoding,
)
from sglang.srt.layers.attention.flashinfer_backend import should_use_tensor_core
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_flashinfer_available

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True


if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        fast_decode_plan,
    )


@dataclass
class DecodeMetadata:
    decode_wrappers: List["BatchDecodeWithPagedKVCacheWrapper"]
    # Dense decode wrapper for layers < start_layer (standard GQA, full KV).
    # None when start_layer == 0.  Carried here (rather than read off
    # self.dense_decode_wrapper) so the eager / capture / replay paths each
    # select the wrapper they planned, without mutating a shared attribute.
    dense_decode_wrapper: Optional["BatchDecodeWithPagedKVCacheWrapper"] = None


@dataclass
class PrefillMetadata:
    prefill_wrappers: List["BatchPrefillWithPagedKVCacheWrapper"]
    use_ragged: bool
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None

# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global_override_indptr_cpu = None


class BlockSparseAttnBackend(AttentionBackend, ABC):
    """Abstract flashinfer-based block-sparse attention backend.

    Subclasses must:

    1. set the sparse-config attributes (``block_size``, ``topk``,
       ``summary_head_dim``, ``start_layer``) by overriding
       :meth:`_init_sparse_config`, which the
       parent ``__init__`` calls *before* allocating flashinfer buffers /
       constructing the indices updaters (which read those attributes);
    2. implement the four abstract summary / selection hooks.
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        init_new_workspace: bool = False,
    ):
        super().__init__()
        self._model_runner = model_runner
        self.device = model_runner.device
        self.prefill_backend = "fa2"
        self.decode_backend = "fa2"

        # Per-TP-rank head counts and GQA constants.
        attention_tp_size = get_attention_tp_size()
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // attention_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            attention_tp_size
        )
        assert self.num_qo_heads % self.num_kv_heads == 0
        self.gqa_group_size = self.num_qo_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.max_context_len = model_runner.model_config.context_len

        # Sparse-config attributes (block_size / topk / summary_head_dim /
        # start_layer), set by the subclass BEFORE we allocate buffers and
        # build the indices updaters.
        self.start_layer = 0
        self._init_sparse_config(model_runner)
        assert hasattr(self, "block_size") and hasattr(self, "topk"), (
            "subclass _init_sparse_config must set self.block_size and self.topk"
        )

        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
        )

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3VLForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
            or "Qwen3VLMoeForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)

        # Allocate workspace buffer
        global global_workspace_buffer
        if global_workspace_buffer is None:
            # different from flashinfer zero_init_global_workspace_buffer
            global_workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
            global_workspace_buffer = torch.empty(
                global_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        if init_new_workspace:
            self.workspace_buffer = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            )
        else:
            self.workspace_buffer = global_workspace_buffer

        # Allocate kv_indptr / kv_last_page_len over the *virtual* batch
        # (max_bs * num_kv_heads), since block-sparse decode folds the kv-head
        # axis into the batch axis.
        max_bs = model_runner.req_to_token_pool.size
        virt_max_bs = max_bs * self.num_kv_heads
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (virt_max_bs + 1,), dtype=torch.int32, device=model_runner.device
            ) # cum_seqlens_kv, virtual batching enabled
        else:
            self.kv_indptr = kv_indptr_buf

        if kv_last_page_len_buf is None:
            self.kv_last_page_len = torch.ones(
                (virt_max_bs,), dtype=torch.int32, device=model_runner.device
            )  # page size = 1, so the last page is always full
        else:
            self.kv_last_page_len = kv_last_page_len_buf

        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        ) # cum_seqlens_q

        # Fixed buffer for the decode KV indices, sized to the virtual-batch
        # upper bound (topk + 1 blocks per virtual request, over max_bs).  Both
        # the eager and the CUDA-graph decode paths write the per-layer expanded
        # indices *in place* here, so the flashinfer wrapper's
        # `_paged_kv_indices_buf` is bound once and never reallocated — the key
        # to making the virtual-batch decode graph-capturable.
        self.decode_kv_indices_buf = torch.empty(
            max_bs * self.num_kv_heads * (self.topk + 1) * self.block_size,
            dtype=torch.int32,
            device=model_runner.device,
        )

        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend=self.prefill_backend,
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend=self.decode_backend,
            use_tensor_cores=self.decode_use_tensor_cores,
        )
        # Pre-bind the eager wrapper's paged-kv-indices buffer to our fixed
        # buffer.  An eager wrapper leaves `_paged_kv_indices_buf` None until its
        # first plan(), so without this the decode updater could not read the
        # buffer off the wrapper before planning.  Binding it here lets both the
        # updater (plan) and forward_decode (per-layer in-place write) reference
        # `decode_wrapper._paged_kv_indices_buf` uniformly — same as the
        # graph wrapper, which gets the buffer at construction — so there is a
        # single, wrapper-centric code path (mirroring the flashinfer backend).
        # eager plan() reassigns `_paged_kv_indices_buf = indices.to(device)`;
        # since we pass this same on-device buffer, it stays bound.
        self.decode_wrapper._paged_kv_indices_buf = self.decode_kv_indices_buf
        # Eager dense decode wrapper for layers < start_layer (standard GQA,
        # full KV).  Only constructed/planned when start_layer > 0
        # (SeerAttention-R).  The CUDA-graph dense wrappers live separately in
        # self.dense_decode_cuda_graph_metadata; this eager wrapper is never
        # reassigned, so the eager fallback path always finds a non-graph
        # wrapper here.
        self.dense_decode_wrapper: Optional[
            "BatchDecodeWithPagedKVCacheWrapper"
        ] = None
        if self.start_layer > 0:
            self.dense_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_tensor_cores=self.decode_use_tensor_cores,
            )
            # Standard (non-virtual) kv_indptr for the dense decode wrapper.
            self.dense_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
            self.dense_kv_last_page_len = torch.ones(
                (max_bs,), dtype=torch.int32, device=model_runner.device
            )
            # Fixed dense kv_indices buffer (full KV range over max_bs), bound
            # once so the dense decode path is graph-safe too.
            self.dense_kv_indices_buf = torch.empty(
                max_bs * self.max_context_len,
                dtype=torch.int32,
                device=model_runner.device,
            )

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata, None] = None

        # CUDA graph state (populated by init_cuda_graph_state /
        # init_forward_metadata_capture_cuda_graph).  Per-batch-size captured
        # decode wrappers are stored so replay re-plans the same wrapper in place.
        self.decode_cuda_graph_metadata: dict = {}
        self.dense_decode_cuda_graph_metadata: dict = {}
        self.disable_cuda_graph_kv_split = False

        # Create indices updaters (read the sparse-config attributes above).
        self.indices_updater_prefill = BlockSparseIndicesUpdaterPrefill(
            model_runner, self
        )
        self.indices_updater_decode = BlockSparseIndicesUpdaterDecode(
            model_runner, self
        )

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _init_sparse_config(self, model_runner: "ModelRunner") -> None:
        """Set sparse-config attributes on ``self`` before buffer allocation.

        Must set at least ``self.block_size`` and ``self.topk``.  May override
        ``self.start_layer`` (default 0) and any backend-specific state (gate
        config, guards, etc.).
        """

    @abstractmethod
    def _build_block_summary_extend(
        self, layer: "RadixAttention", k: torch.Tensor, forward_batch: ForwardBatch, **kwargs
    ) -> None:
        """Build/update block summaries for the new tokens during extend."""

    @abstractmethod
    def _update_block_summary_decode(
        self, layer: "RadixAttention", k: torch.Tensor, forward_batch: ForwardBatch, **kwargs
    ) -> None:
        """Update block summaries for the new decode token."""

    @abstractmethod
    def _select_active_blocks(
        self, layer: "RadixAttention", q: torch.Tensor, forward_batch: ForwardBatch, **kwargs
    ) -> torch.Tensor:
        """Select active *previous* blocks per (request, kv-head).

        Returns ``[bsz, num_kv_heads, topk]`` int32, ``-1`` padded, ascending.
        Must NOT include the current/partial block (appended downstream).
        """

    def _is_sparse_layer(self, layer: "RadixAttention", **kwargs) -> bool:
        """Whether this layer runs block-sparse (vs dense) for this forward.

        Default: every layer at or beyond ``start_layer`` is sparse.  Subclasses
        may further gate on per-forward inputs (e.g. SeerAttention-R also
        requires the gate tensors to be present in ``kwargs``).
        """
        return layer.layer_id >= self.start_layer

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrapper=self.decode_wrapper,
                fixed_split_size=None,
                disable_split_kv=False,
            )
            # Plan the dense decode wrapper (layers < start_layer) over the full
            # KV range.  Only needed when start_layer > 0.
            if self.dense_decode_wrapper is not None:
                self._plan_dense_decode(
                    self.dense_decode_wrapper,
                    forward_batch.seq_lens.size(0),
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                )
            self.forward_metadata = DecodeMetadata(
                [self.decode_wrapper], dense_decode_wrapper=self.dense_decode_wrapper
            )
        else:
            # Dense paged prefill (sparsity only applies to decode).
            prefix_lens = forward_batch.extend_prefix_lens
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrapper=self.prefill_wrapper,
            )
            self.forward_metadata = PrefillMetadata(
                [self.prefill_wrapper],
                use_ragged=False,
                extend_no_prefix=False,
            )

    def _plan_dense_decode(
        self,
        dense_wrapper: "BatchDecodeWithPagedKVCacheWrapper",
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """Plan a standard GQA decode wrapper over the full KV range.

        Used for layers ``< start_layer`` (SeerAttention-R), which run dense
        decode while the rest of the network runs block-sparse.  Writes the KV
        indices into the fixed ``self.dense_kv_indices_buf`` (bound once) so the
        path is identical in eager and graph modes.  The caller supplies the
        target wrapper (the eager wrapper in eager mode, or the per-bs captured
        graph wrapper during capture/replay) and the raw (bs, req_pool_indices,
        seq_lens) — there is a single planning code path for all three modes.

        FUTURE (sparse prefill): this is the dense-plan for decode layers
        ``< start_layer``.  When prefill is made block-sparse for ``>=
        start_layer`` layers, the natural refactor is to split planning along
        the **dense vs sparse** axis rather than the current **prefill vs
        decode** axis: a dense planner (this method + the dense paged-prefill
        plan in ``BlockSparseIndicesUpdaterPrefill``, serving non-sparse prefill
        layers AND decode ``< start_layer``) and a sparse planner (the
        virtual-batch decode plan in ``BlockSparseIndicesUpdaterDecode`` + a new
        sparse-prefill plan, serving sparse layers in both modes).  Do NOT fold
        this into the decode updater before then — it would have to be pulled
        back out.  See the matching notes on ``forward_extend`` and the two
        updater classes.
        """
        kv_indptr = self.dense_kv_indptr
        kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = self.dense_kv_indices_buf
        req_to_token = self._model_runner.req_to_token_pool.req_to_token
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            req_to_token.shape[1],
        )
        dense_wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            self.dense_kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,  # page size
            data_type=self._model_runner.kv_cache_dtype,
            q_data_type=self._model_runner.dtype,
            non_blocking=True,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ------------------------------------------------------------------
    # CUDA graph
    #
    # The virtual-batch block-sparse decode is graph-capturable because the
    # per-layer KV-index expansion writes into fixed pre-allocated buffers
    # (`decode_kv_indices_buf`, and for start_layer>0 the dense
    # `dense_kv_indices_buf`).  These hooks mirror the standard flashinfer
    # backend: capture builds `use_cuda_graph=True` wrappers bound to those fixed
    # buffers and swaps `begin_forward` for `fast_decode_plan`; replay re-plans in
    # place.  A subclass whose per-step summary update is not graph-safe (e.g.
    # one that needs a `.item()` host sync) keeps graph disabled at the
    # model-runner level; MoBA and SeerAttention-R use fixed-grid, sync-free
    # decode kernels.
    # ------------------------------------------------------------------
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # The fixed KV-index buffers are already allocated in __init__ (sized to
        # the request-pool upper bound, which covers any graph capture batch
        # size), so nothing more to allocate here.  Present for interface
        # parity with the flashinfer backend.
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
    ):
        assert (
            forward_mode.is_decode_or_idle()
        ), "block-sparse backends only capture decode/idle graphs"

        virt_bs = bs * self.num_kv_heads
        # Virtual-batch sparse decode wrapper bound to the fixed buffers.
        decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend=self.decode_backend,
            use_cuda_graph=True,
            use_tensor_cores=self.decode_use_tensor_cores,
            paged_kv_indptr_buffer=self.kv_indptr[: virt_bs + 1],
            paged_kv_indices_buffer=self.decode_kv_indices_buf,
            paged_kv_last_page_len_buffer=self.kv_last_page_len[:virt_bs],
        )
        seq_lens_sum = int(seq_lens.sum().item())
        self.indices_updater_decode.update(
            req_pool_indices,
            seq_lens,
            seq_lens.cpu(),
            seq_lens_sum,
            decode_wrapper=decode_wrapper,
            fixed_split_size=None,
            disable_split_kv=self.disable_cuda_graph_kv_split,
        )
        # Swap begin_forward -> fast_decode_plan for graph-safe in-place replanning.
        decode_wrapper.begin_forward = partial(fast_decode_plan, decode_wrapper)
        self.decode_cuda_graph_metadata[bs] = decode_wrapper

        # Dense decode wrapper (layers < start_layer), if any.  The captured
        # wrapper is stored only in dense_decode_cuda_graph_metadata[bs] and
        # carried in forward_metadata — self.dense_decode_wrapper (the eager
        # wrapper) is never overwritten, so the eager fallback path stays valid.
        dense_wrapper = None
        if self.dense_decode_wrapper is not None:
            dense_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=self.dense_kv_indptr[: bs + 1],
                paged_kv_indices_buffer=self.dense_kv_indices_buf,
                paged_kv_last_page_len_buffer=self.dense_kv_last_page_len[:bs],
            )
            self._plan_dense_decode(dense_wrapper, bs, req_pool_indices, seq_lens)
            dense_wrapper.begin_forward = partial(fast_decode_plan, dense_wrapper)
            self.dense_decode_cuda_graph_metadata[bs] = dense_wrapper
        self.forward_metadata = DecodeMetadata(
            [decode_wrapper], dense_decode_wrapper=dense_wrapper
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        assert (
            forward_mode.is_decode_or_idle()
        ), "block-sparse backends only replay decode/idle graphs"

        decode_wrapper = self.decode_cuda_graph_metadata[bs]
        self.indices_updater_decode.update(
            req_pool_indices[:bs],
            seq_lens[:bs],
            seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
            seq_lens_sum,
            decode_wrapper=decode_wrapper,
            fixed_split_size=None,
            disable_split_kv=self.disable_cuda_graph_kv_split,
        )
        dense_wrapper = None
        if self.dense_decode_wrapper is not None:
            dense_wrapper = self.dense_decode_cuda_graph_metadata[bs]
            self._plan_dense_decode(
                dense_wrapper, bs, req_pool_indices[:bs], seq_lens[:bs]
            )
        # Refresh forward_metadata so it reflects this bs's captured wrappers.
        # (Replay re-runs the captured graph rather than forward_decode, so this
        # is for state consistency, not correctness of the replayed kernels.)
        self.forward_metadata = DecodeMetadata(
            [decode_wrapper], dense_decode_wrapper=dense_wrapper
        )

    # ------------------------------------------------------------------
    # Extend (dense) + summary build
    # ------------------------------------------------------------------
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        if layer.is_cross_attention:
            raise NotImplementedError(
                "Cross attention is not supported in block-sparse backends"
            )

        cache_loc = forward_batch.out_cache_loc
        q = q.contiguous()

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
            if self._is_sparse_layer(layer, **kwargs):
                self._build_block_summary_extend(layer, k, forward_batch, **kwargs)

        # Prefill is currently DENSE for every layer (sparsity applies only to
        # decode); the per-block gate summaries built above are consumed later at
        # decode time.  FUTURE (sparse prefill): for ``_is_sparse_layer`` layers
        # this will branch to a block-sparse prefill (gate-select key blocks per
        # query block -> expand kv_indices -> sparse-prefill wrapper), mirroring
        # the decode sparse path.  When that lands, planning should be reorganised
        # along the dense/sparse axis (see ``_plan_dense_decode``'s docstring),
        # so the dense branch here shares a planner with decode ``< start_layer``
        # and the sparse branch shares one with sparse decode.
        o = self.prefill_wrapper.forward(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            causal=not layer.is_cross_attention,
            sm_scale=layer.scaling,
            window_left=-1,
            logits_soft_cap=layer.logit_cap,
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    # ------------------------------------------------------------------
    # Decode (block-sparse, virtual batching)
    # ------------------------------------------------------------------
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        if layer.is_cross_attention:
            raise NotImplementedError(
                "Cross attention is not supported in block-sparse backends"
            )

        cache_loc = forward_batch.out_cache_loc
        bsz = forward_batch.seq_lens.size(0)
        token_to_kv_pool = forward_batch.token_to_kv_pool
        H = self.num_kv_heads
        is_sparse = self._is_sparse_layer(layer, **kwargs)

        # Step 1: persist KV and (for sparse layers) update block summary.
        if k is not None:
            assert v is not None
            if save_kv_cache:
                token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
            if is_sparse:
                self._update_block_summary_decode(layer, k, forward_batch, **kwargs)

        q_3d = q.contiguous().view(bsz, layer.tp_q_head_num, layer.head_dim)

        # Non-sparse layers (layer_id < start_layer, or missing routing inputs)
        # run dense decode (standard GQA, full KV).  Use the wrapper carried in
        # forward_metadata (eager wrapper in eager mode, the per-bs captured
        # wrapper during capture) — never self.dense_decode_wrapper, which is
        # only the eager wrapper and would be wrong inside a graph capture.
        if not is_sparse:
            dense_wrapper = self.forward_metadata.dense_decode_wrapper
            o = dense_wrapper.forward(
                q_3d,
                token_to_kv_pool.get_kv_buffer(layer.layer_id),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            return o.view(bsz, layer.tp_q_head_num * layer.head_dim)

        # Step 2: select active blocks (subclass routing) — [bsz, H, topk].
        # query-to-block scoring -> compute threshold -> compute active block ids
        active_block_ids = self._select_active_blocks(
            layer, q_3d, forward_batch, **kwargs
        )

        # Step 3: expand to virtual KV indices (representation-agnostic; the
        # current/partial block is appended by position here).  The decode
        # wrapper comes from forward_metadata: the eager wrapper in eager mode,
        # or the per-bs captured wrapper during graph capture/replay.  Either
        # way it is bound to `self.decode_kv_indices_buf`, so we expand the
        # indices *in place* into that fixed buffer — no per-layer rebind, which
        # is what makes this graph-safe.
        decode_wrapper = self.forward_metadata.decode_wrappers[0]
        block_inds_to_kv_inds_for_decoding(
            active_block_ids,
            forward_batch.seq_lens,
            forward_batch.req_pool_indices,
            self._model_runner.req_to_token_pool.req_to_token,
            block_size=self.block_size,
            num_kv_heads=H,
            topk=self.topk,
            # Reuse the virtual-batch kv_indptr built once this step in
            # init_forward_metadata instead of recomputing it per layer.
            kv_indptr=self.kv_indptr,
            out=decode_wrapper._paged_kv_indices_buf,
        )

        # Step 4: reshape to virtual kv layout and run attention.
        G = self.gqa_group_size
        # Q: [bsz, num_q_heads, d] -> [bsz, H, G, d] -> [bsz*H, G, d]
        q_virtual = q_3d.view(bsz, H, G, self.head_dim).reshape(
            bsz * H, G, self.head_dim
        )
        # (pool_size, num_kv_heads_tp, d) -> (pool_size*num_kv_heads_tp, 1, d)
        k_buf, v_buf = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        pool_size = k_buf.size(0)
        k_buf_virtual = k_buf.view(pool_size * H, 1, self.head_dim)
        v_buf_virtual = v_buf.view(pool_size * H, 1, self.head_dim)

        o = decode_wrapper.forward(
            q_virtual,
            (k_buf_virtual, v_buf_virtual),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        # output: [bsz*H, G, d] -> [bsz, num_q_heads*d]
        return o.reshape(bsz, layer.tp_q_head_num * layer.head_dim)


class BlockSparseIndicesUpdaterDecode:
    """Plans the virtual-batch decode wrapper for block-sparse decode.

    Builds the virtual-batch ``kv_indptr`` over a *fixed* sparse budget
    (``topk`` previous blocks + the current partial block) so the plan is done
    once per step and shared across layers; only ``kv_indices`` is recomputed
    per layer (in ``forward_decode``).

    FUTURE (sparse prefill): this is the *sparse* planner for decode.  When
    prefill is made block-sparse, the sparse-prefill plan belongs here (or in a
    shared sparse planner) alongside this decode plan — i.e. group planners by
    dense/sparse, not by prefill/decode.  See ``_plan_dense_decode``'s docstring.
    """

    def __init__(self, model_runner: "ModelRunner", attn_backend: BlockSparseAttnBackend):
        self.num_qo_heads = attn_backend.num_qo_heads
        self.num_kv_heads = attn_backend.num_kv_heads
        self.head_dim = attn_backend.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.attn_backend = attn_backend

        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.block_size = attn_backend.block_size
        self.topk = attn_backend.topk
        self.gqa_group_size = attn_backend.gqa_group_size

    def _compute_sparse_kv_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        current_block_id = (seq_lens - 1) // self.block_size  # [bsz]
        num_prev_blocks = current_block_id
        actual_topk = torch.clamp(num_prev_blocks, max=self.topk)  # [bsz]
        cur_block_len = seq_lens - current_block_id * self.block_size  # [bsz]
        sparse_kv_lens = actual_topk * self.block_size + cur_block_len  # [bsz]
        return sparse_kv_lens

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrapper: "BatchDecodeWithPagedKVCacheWrapper",
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # TODO: kernelize kv_indptr computation?
        sparse_kv_lens = self._compute_sparse_kv_lens(seq_lens)  # [bsz]
        # expand to virtual batch
        virt_seq_lens = sparse_kv_lens.repeat_interleave(
            self.num_kv_heads
        )  # [bsz * num_kv_heads]
        virt_bsz = virt_seq_lens.size(0)  # bsz * num_kv_heads
        # kv_indptr: physical size [max_virt_bs + 1], valid size [virt_bsz + 1]
        kv_indptr = self.kv_indptr
        kv_indptr[1 : virt_bsz + 1] = torch.cumsum(virt_seq_lens, dim=0)
        kv_indptr = kv_indptr[: virt_bsz + 1]

        # NOTE: for block sparse attention, kv_indices is computed per-layer in
        # forward_decode; the plan only needs kv_indptr (segment offsets), not
        # the index values.  Take the buffer off the wrapper (mirroring the
        # flashinfer backend's `kv_indices = wrapper._paged_kv_indices_buf`) so
        # plan and the per-layer in-place expansion reference the exact same
        # tensor — eager and graph wrappers both have it bound to the backend's
        # fixed `decode_kv_indices_buf`, so there is one code path for both.
        kv_indices = decode_wrapper._paged_kv_indices_buf

        global global_override_indptr_cpu
        if seq_lens_cpu is not None:
            sparse_kv_lens_cpu = self._compute_sparse_kv_lens(seq_lens_cpu)
            virt_seq_lens_cpu = sparse_kv_lens_cpu.repeat_interleave(
                self.num_kv_heads
            )  # [bsz * num_kv_heads]
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
            global_override_indptr_cpu[0] = 0
            global_override_indptr_cpu[1 : virt_bsz + 1] = torch.cumsum(
                virt_seq_lens_cpu, dim=0
            )

        # Check if this specific wrapper's begin_forward has been replaced with
        # fast_decode_plan (a partial function with fast_decode_plan as func).
        wrapper_uses_fast_decode_plan = (
            hasattr(decode_wrapper.begin_forward, "func")
            and decode_wrapper.begin_forward.func == fast_decode_plan
        )
        plan_kwargs = dict(
            data_type=self.data_type,
            q_data_type=self.q_data_type,
            non_blocking=True,
            fixed_split_size=fixed_split_size,
            disable_split_kv=(
                disable_split_kv if disable_split_kv is not None else False
            ),
        )
        if wrapper_uses_fast_decode_plan:  # only for cudagraph capturing stage
            plan_kwargs["global_override_indptr_cpu"] = global_override_indptr_cpu

        decode_wrapper.begin_forward(
            kv_indptr,  # [virt_bsz + 1]
            kv_indices,  # [virt_seq_lens_sum]
            self.kv_last_page_len[:virt_bsz],  # page_size=1, never updated
            self.gqa_group_size,  # num_qo_heads per virtual request
            1,  # num_kv_heads per virtual request
            self.head_dim,
            1,  # page size
            **plan_kwargs,
        )


class BlockSparseIndicesUpdaterPrefill:
    """Plans the single paged prefill wrapper for dense (non-sparse) prefill.

    Block-sparse backends perform *dense* prefill — sparsity is applied only
    during decode.  This updater builds the standard paged ``kv_indptr`` /
    ``kv_indices`` / ``qo_indptr`` over the full (prefix + extend) KV range and
    calls the wrapper's ``begin_forward``.

    FUTURE (sparse prefill): this is the *dense* planner for prefill.  It shares
    its shape with the dense decode plan (``_plan_dense_decode``); when prefill
    is made block-sparse for ``>= start_layer`` layers, the sparse-prefill plan
    should live with the sparse decode planner, and this dense planner should
    serve only non-sparse prefill layers — i.e. group by dense/sparse, not by
    prefill/decode.  See ``_plan_dense_decode``'s docstring.
    """

    def __init__(self, model_runner: "ModelRunner", attn_backend: BlockSparseAttnBackend):
        self.num_qo_heads = attn_backend.num_qo_heads
        self.num_kv_heads = attn_backend.num_kv_heads
        self.head_dim = attn_backend.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.attn_backend = attn_backend

        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrapper: "BatchPrefillWithPagedKVCacheWrapper",
    ):
        bs = len(seq_lens)

        # Paged KV covers the full sequence (prefix + extend) for every request.
        paged_kernel_lens = seq_lens
        paged_kernel_lens_sum = seq_lens_sum

        kv_indptr = self.kv_indptr
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            paged_kernel_lens_sum + 256,
            dtype=torch.int32,
            device=req_pool_indices.device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.shape[1],
        )

        # qo_indptr spans the new (extend) tokens only.
        qo_indptr = self.qo_indptr
        qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        plan_kwargs = dict(
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=None,
            non_blocking=True,
        )

        prefill_wrapper.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,  # page size
            **plan_kwargs,
        )
