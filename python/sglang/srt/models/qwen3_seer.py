# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Qwen3 with SeerAttention-R decode-sparse attention gates.

This variant of the Qwen3 model attaches a per-layer :class:`AttnGate` (the
trained SeerAttention-R gate) to each attention layer.  During decode, the gate
selects which KV blocks each query attends to; the heavy lifting (rolling/
summary cache maintenance, scoring, block-sparse decode) lives in
:class:`~sglang.srt.layers.attention.blocksparse.seer_attn.backend.SeerAttnBackend`.

The model's job is to:

* own the gate modules (so their weights load and live on-device), and
* pass the gate plus the *pre-RoPE, post-qknorm* Q/K and RoPE tables to the
  attention backend via the ``RadixAttention`` ``**kwargs`` channel.

Weight loading composes a frozen base model (``config.base_model``) with the
distilled gate weights (``attn_gate_weights.pth`` in the checkpoint dir), which
is how SeerAttention-R ships its models.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Iterable, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from torch import nn

import sglang.srt.models.qwen3 as qwen3_mod
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate import AttnGate
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from sglang.srt.models.qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3ForCausalLM
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


# Cache of gate RoPE inverse frequencies, keyed by (rope_dim, rope_theta,
# device).  Every attention layer's gate uses identical RoPE parameters, so a
# single [rope_dim//2] inv_freq vector is shared across all layers.  cos/sin are
# computed in-kernel (NeoX, fp32) from these, so no per-position table is built.
_GATE_ROPE_CACHE: dict = {}


def _get_gate_rope_tables(rope_dim, rope_theta, max_pos, device, dtype):
    """Return cached NeoX gate RoPE inverse frequencies.

    Matches HuggingFace/reference RoPE: ``inv_freq`` over half the dim.  Used by
    the AttnGate (whose RoPE dimension equals ``gate_hidden_size``); the Triton
    kernels compute ``cos``/``sin`` in fp32 from ``inv_freq`` at the relevant
    token positions (Q at the current position, summary K at block positions),
    so no ``[max_pos, rope_dim]`` table is materialised.

    Returns ``(None, None, inv_freq)`` — the leading slots are kept for
    signature stability; only ``inv_freq`` (``[rope_dim//2]`` fp32) is used.
    """
    key = (rope_dim, float(rope_theta), str(device))
    cached = _GATE_ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    tables = (None, None, inv_freq)
    _GATE_ROPE_CACHE[key] = tables
    return tables


class SeerAttnQwen3Attention(Qwen3Attention):
    """Qwen3 attention augmented with a SeerAttention-R gate.

    Reuses the base ``forward_prepare_native`` to obtain post-qknorm,
    post-RoPE q/k/v, but *also* captures the post-qknorm, *pre-RoPE* q/k (the
    gate inputs) and forwards everything the backend needs through kwargs.
    """

    def __init__(self, *args, seer_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert seer_cfg is not None
        self.seer_cfg = seer_cfg
        self.attn_gate = AttnGate(
            block_size=seer_cfg.block_size,
            model_hidden_size=self.head_dim,
            gate_hidden_size=seer_cfg.gate_hidden_size,
            num_k_head=self.num_kv_heads,
            num_q_head=self.num_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Replicate forward_prepare_native, but keep the pre-RoPE q/k.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        # Pre-RoPE, post-qknorm gate inputs.
        num_tokens = q.shape[0]
        q_nope = q.view(num_tokens, self.num_heads, self.head_dim)
        k_nope = k.view(num_tokens, self.num_kv_heads, self.head_dim)

        q_rope, k_rope = self.rotary_emb(positions, q, k)

        # The gate RoPE (Q at the current position, summary K at block
        # positions) is computed in-kernel from inv_freq, so only the
        # position-independent inverse frequencies are passed.
        inv_freq = self._gate_rope_inv_freq(hidden_states.device, q.dtype)

        attn_output = self.attn(
            q_rope,
            k_rope,
            v,
            forward_batch,
            seer_gate=self.attn_gate,
            seer_q_nope=q_nope,
            seer_k_nope=k_nope,
            seer_rope_inv_freq=inv_freq,
        )
        output, _ = self.o_proj(attn_output)
        return output

    def _gate_rope_inv_freq(self, device, dtype):
        # Gate RoPE inverse frequencies (NeoX, gate_hidden_size-dim), shared
        # across all layers via a module-level cache keyed on
        # (rope_dim, theta, max_pos, device, dtype).
        rope_dim = self.seer_cfg.gate_hidden_size
        _, _, inv_freq = _get_gate_rope_tables(
            rope_dim, self.rope_theta, self.max_position_embeddings, device, dtype
        )
        return inv_freq


class SeerAttnQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config=None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        )
        # Replace the plain attention with the gate-augmented variant.
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = SeerAttnQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            seer_cfg=config.seer_attn_cfg,
        )


def _shard_gate_weight(
    name: str,
    w: torch.Tensor,
    attn_tp_rank: int,
    attn_tp_size: int,
) -> torch.Tensor:
    """Shard a full-head gate weight to this attention-TP rank.

    The released ``attn_gate_weights.pth`` holds the *full* head count, while
    the model's gate parameters are built with the TP-local head counts (the
    AttnGate is constructed from ``SeerAttnQwen3Attention.num_kv_heads`` /
    ``num_heads``, which :class:`Qwen3Attention` has already divided by
    ``attn_tp_size``).  So under TP>1 the per-head gate projections must be
    sliced along their kv-head axis (dim 0) before loading; the head-invariant
    norm weights are replicated.

    Layouts (see ``attn_gate.py``):

    * ``attngate_linear_q.weight`` ``[num_kv_head, gqa_group, head_dim, gate_dim]``
      -> slice dim 0.  The GQA-group axis (dim 1) is ``num_q_head //
      num_kv_head``, which is TP-invariant in the kv-head *partition* regime
      (both numerator and denominator are divided by ``attn_tp_size``), so it
      rides along with its owning kv head untouched -- mirroring how
      ``Qwen3Attention`` shards Q heads by their kv group.
    * ``attngate_linear_k.weight`` ``[num_kv_head, head_dim*3, gate_dim]``
      -> slice dim 0.
    * ``attngate_qnorm.weight`` / ``attngate_knorm.weight`` ``[gate_dim]``
      -> replicate (not a function of heads).

    Only the kv-head *partition* regime (``num_kv_heads >= attn_tp_size``) is
    supported; the replicate regime is rejected upstream in ``__init__``.
    """
    if attn_tp_size == 1:
        return w
    if name.endswith(("qnorm.weight", "knorm.weight")):
        # gate_dim vector, head-independent -> every rank holds the full copy.
        return w
    if name.endswith(("linear_q.weight", "linear_k.weight")):
        total_kv_heads = w.shape[0]
        assert total_kv_heads % attn_tp_size == 0, (
            f"gate weight {name!r} has {total_kv_heads} kv heads, not divisible "
            f"by attention tp_size={attn_tp_size}; the kv-head replicate regime "
            "is unsupported (rejected in __init__)."
        )
        local = total_kv_heads // attn_tp_size
        return w[attn_tp_rank * local : (attn_tp_rank + 1) * local]
    # Unknown gate parameter: load whole (no head axis assumed).  The set of
    # gate params is fixed by AttnGate, so this is defensive only.
    return w


class SeerAttnQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 + SeerAttention-R gates.

    Note: relies on :class:`~sglang.srt.models.qwen3.Qwen3Model` instantiating
    decoder layers through ``decoder_layer_type``.  We swap that type here.
    """

    def __init__(self, config, quant_config=None, prefix: str = ""):
        # The AttnGate forward is TP-safe (it is built from the TP-local head
        # counts via SeerAttnQwen3Attention) and _load_gate_weights shards the
        # full-head checkpoint along the kv-head axis (see _shard_gate_weight).
        # Only the kv-head *replicate* regime (more TP ranks than kv heads) is
        # unsupported: there the gate's GQA-group axis would also have to be
        # repartitioned and the per-kv-head slice no longer maps one-to-one, so
        # reject it explicitly rather than load wrong weights.
        attn_tp_size = get_attention_tp_size()
        total_num_kv_heads = config.num_key_value_heads
        if attn_tp_size > 1 and total_num_kv_heads < attn_tp_size:
            raise NotImplementedError(
                "SeerAttention-R AttnGate does not support the kv-head replicate "
                f"regime (num_key_value_heads={total_num_kv_heads} < attention "
                f"tp_size={attn_tp_size}). Use a tp_size that divides the kv-head "
                "count (e.g. <= num_key_value_heads)."
            )
        if attn_tp_size > 1 and total_num_kv_heads % attn_tp_size != 0:
            raise NotImplementedError(
                f"num_key_value_heads={total_num_kv_heads} is not divisible by "
                f"attention tp_size={attn_tp_size}; cannot evenly shard the "
                "AttnGate kv-head axis."
            )

        # Attach the resolved seer config (built in ModelConfig) onto the HF
        # config so the decoder layers can read it.  When ModelConfig isn't the
        # caller (e.g. unit tests), fall back to reading seerattn_* fields.
        if not hasattr(config, "seer_attn_cfg"):
            config.seer_attn_cfg = _seer_cfg_from_hf(config)
        # Monkeypatch the decoder layer type used by Qwen3Model/Qwen2Model.
        self._orig_decoder_layer = qwen3_mod.Qwen3DecoderLayer
        qwen3_mod.Qwen3DecoderLayer = SeerAttnQwen3DecoderLayer
        try:
            super().__init__(config, quant_config=quant_config, prefix=prefix)
        finally:
            qwen3_mod.Qwen3DecoderLayer = self._orig_decoder_layer

        self._gate_path = getattr(config, "seer_attn_gate_path", None)
        self._base_model = getattr(config, "base_model", None)

    # ------------------------------------------------------------------
    # Weight loading: frozen base model + distilled gate weights.
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The provided ``weights`` iterator scans the AttnGates checkpoint dir,
        # which contains ONLY ``attn_gate_weights.pth`` (no base safetensors).
        # We therefore IGNORE it (consuming it would trigger a "no model
        # weights found" scan) and instead load:
        #   1. the frozen base Qwen3 weights from ``config.base_model``, and
        #   2. the distilled gate weights from ``attn_gate_weights.pth``.
        del weights  # intentionally unused (see above)
        self._load_base_weights()
        self._load_gate_weights()

    def _load_base_weights(self):
        if not self._base_model:
            raise ValueError(
                "SeerAttnQwen3 requires config.base_model to locate the frozen "
                "base model weights (the AttnGates checkpoint ships only the "
                "gate weights)."
            )
        loader = DefaultModelLoader(LoadConfig())
        hf_folder, hf_weights_files, use_safetensors = loader._prepare_weights(
            self._base_model, revision=None, fall_back_to_pt=True
        )
        _it = safetensors_weights_iterator if use_safetensors else pt_weights_iterator
        # Reuse the base Qwen3 loader for stacked-param mapping.
        super().load_weights(_it(hf_weights_files))

    def _load_gate_weights(self):
        params_dict = dict(self.named_parameters())
        loaded = 0
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        def _try_load(name, w):
            nonlocal loaded
            if "attn_gate" not in name:
                return
            key = name if name.startswith(("model.", "lm_head")) else "model." + name
            if key in params_dict:
                param = params_dict[key]
                # The checkpoint carries full-head gate weights; shard to this
                # attention-TP rank before loading (no-op when tp_size == 1).
                w = _shard_gate_weight(name, w, attn_tp_rank, attn_tp_size)
                wl = getattr(param, "weight_loader", default_weight_loader)
                wl(param, w)
                loaded += 1

        # Gate weights from attn_gate_weights.pth in the checkpoint dir.
        if self._gate_path:
            pth = os.path.join(self._gate_path, "attn_gate_weights.pth")
            if not os.path.exists(pth):
                pth = hf_hub_download(
                    repo_id=self._gate_path, filename="attn_gate_weights.pth"
                )
            sd = torch.load(pth, map_location="cpu")
            for name, w in sd.items():
                _try_load(name, w)

        logger.info(f"SeerAttnQwen3: loaded {loaded} gate tensors.")


def _seer_cfg_from_hf(config):
    return SimpleNamespace(
        block_size=int(config.seerattn_gate_block_size),
        gate_hidden_size=int(config.seerattn_gate_hidden_size),
        sparsity_method=getattr(config, "seerattn_sparsity_method", "token_budget"),
        threshold=float(getattr(config, "seerattn_threshold", 0.0)),
        token_budget=int(getattr(config, "seerattn_token_budget", 4096)),
        start_layer=int(getattr(config, "seerattn_start_layer", 0)),
    )


EntryClass = SeerAttnQwen3ForCausalLM
