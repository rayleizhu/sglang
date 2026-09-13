# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""SeerAttention-R AttnGate module (inference path) for SGLang.

Port of the AttnGate used by SeerAttention-R
(https://github.com/microsoft/SeerAttention, arXiv:2506.08889).  The gate is a
small trained module that, at *decode* time, predicts which KV *blocks* each
query should attend to.  Only the gate weights are distilled/shipped; they are
composed onto a frozen base model.

This is specialised to the configuration used by **every released Qwen3 decode
AttnGate** checkpoint, so the reference's configurable branches are dropped:

* Q head pooling = ``Qproj`` (learned per-(K-head,GQA-group) projection),
* K sequence pooling = ``Kmaxminavg`` (max+min+avg pooled, then concatenated),
* qk-norm = on, RoPE = on (NeoX style, gate dim == head dim).

Cache maintenance (rolling pre-RoPE K + per-block compressed-K summary) and
RoPE application live in the attention backend, because they depend on SGLang's
paged KV layout.  Tensor layout here is SGLang's flat decode layout
``[num_tokens, num_heads, head_dim]`` (one token per request during decode).

Parameter names (``attngate_linear_q``, ``attngate_linear_k``,
``attngate_qnorm``, ``attngate_knorm``) match the reference so that the released
``attn_gate_weights.pth`` loads without remapping.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# K-branch pooling is fixed to max+min+avg ("Kmaxminavg") and concatenated in
# this order, matching the trained ``attngate_linear_k`` weight layout.
K_POOL_DUP = 3


def _min_pool(x, kernel_size, stride):
    return -F.max_pool3d(-x, kernel_size=kernel_size, stride=stride, ceil_mode=True)


class RMSNorm(nn.Module):
    """Plain RMSNorm matching the reference ``seer_attn.modules.common.RMSNorm``.

    Kept local (rather than reusing SGLang's fused RMSNorm) so numerics match
    the reference bit-for-bit: fp32 variance, weight applied after cast-back.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HeadPoolingLinear(nn.Module):
    """Q-branch projection that pools GQA query heads into K-head groups (``Qproj``).

    Weight shape ``[num_k_head, gqa_group_size, model_hidden_size, gate_hidden_size]``.
    Input ``[seq, num_q_head, model_hidden_size]`` -> ``[seq, num_k_head, gate_hidden_size]``:
    the GQA group of query heads belonging to each K head is folded in via the
    per-group weight.
    """

    def __init__(
        self,
        num_k_head: int,
        gqa_group_size: int,
        model_hidden_size: int,
        gate_hidden_size: int,
    ):
        super().__init__()
        self.num_k_head = num_k_head
        self.gqa_group_size = gqa_group_size
        self.weight = nn.Parameter(
            torch.empty(num_k_head, gqa_group_size, model_hidden_size, gate_hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq, num_q_head, model_hidden_size]
        x = x.view(x.shape[0], self.num_k_head, self.gqa_group_size, x.shape[2])
        return torch.einsum("skgi,kgio->sko", x, self.weight)


class MultiHeadLinear(nn.Module):
    """Per-head linear projection (the K branch, ``attngate_linear_k``).

    Weight ``[num_head, in_channel, hidden_size]``;
    input ``[seq, num_head, in_channel]`` -> ``[seq, num_head, hidden_size]``.
    """

    def __init__(self, in_channel_size: int, hidden_size: int, num_head: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_head, in_channel_size, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("shi,hio->sho", x, self.weight)


class AttnGate(nn.Module):
    """SeerAttention-R AttnGate (inference-only math, Qproj + Kmaxminavg + qk-norm).

    Responsibilities (cache + RoPE done by the backend):

    * :meth:`pool_block_keys` — max/min/avg pool a block of raw (pre-RoPE,
      post-k_norm) keys and concatenate -> input to the K linear.
    * :meth:`compress_k` — ``attngate_linear_k`` + knorm -> per-block
      compressed-K vectors of width ``gate_hidden_size`` (RoPE applied by caller).
    * :meth:`project_q` — ``attngate_linear_q`` + qnorm -> per-(K-head) gate
      queries of width ``gate_hidden_size`` (RoPE applied by caller).
    * :attr:`scale` — the gate's softmax scale.
    """

    def __init__(
        self,
        block_size: int,
        model_hidden_size: int,  # == head_dim of the base model
        gate_hidden_size: int,
        num_k_head: int,
        num_q_head: int,
    ):
        super().__init__()
        self.block_size = block_size
        self.gate_hidden_size = gate_hidden_size
        self.num_k_head = num_k_head
        self.num_q_head = num_q_head
        self.gqa_group_size = num_q_head // num_k_head

        self.attngate_linear_q = HeadPoolingLinear(
            num_k_head, self.gqa_group_size, model_hidden_size, gate_hidden_size
        )
        self.attngate_linear_k = MultiHeadLinear(
            model_hidden_size * K_POOL_DUP, gate_hidden_size, num_k_head
        )
        self.attngate_qnorm = RMSNorm(gate_hidden_size, eps=1e-6)
        self.attngate_knorm = RMSNorm(gate_hidden_size, eps=1e-6)

        self.scale = 1.0 / math.sqrt(gate_hidden_size)

    # ------------------------------------------------------------------
    # K branch
    # ------------------------------------------------------------------
    def pool_block_keys(self, k_block: torch.Tensor) -> torch.Tensor:
        """Max/min/avg pool keys within blocks and concatenate the features.

        Args:
            k_block: ``[N, block_len, num_k_head, head_dim]`` raw (pre-RoPE,
                post-k_norm) keys.  ``block_len`` may be < ``block_size`` for a
                partial final block — pooling still covers the whole length.
        Returns:
            ``[N, num_k_head, head_dim * 3]`` (max | min | avg) pooled keys.
        """
        n, block_len, hk, d = k_block.shape
        x = k_block.unsqueeze(1)  # [N, 1, block_len, hk, d]
        ksz = [block_len, 1, 1]
        pooled = [
            F.max_pool3d(x, kernel_size=ksz, stride=ksz, ceil_mode=True),
            _min_pool(x, kernel_size=ksz, stride=ksz),
            F.avg_pool3d(x, kernel_size=ksz, stride=ksz, ceil_mode=True),
        ]
        return torch.cat([p.reshape(n, hk, d) for p in pooled], dim=-1)

    def compress_k(self, k_pooled: torch.Tensor) -> torch.Tensor:
        """Project pooled keys to gate space + knorm (no RoPE; caller applies it).

        ``[N, num_k_head, head_dim*3]`` -> ``[N, num_k_head, gate_hidden_size]``.
        """
        return self.attngate_knorm(self.attngate_linear_k(k_pooled))

    # ------------------------------------------------------------------
    # Q branch
    # ------------------------------------------------------------------
    def project_q(self, q: torch.Tensor) -> torch.Tensor:
        """Project decode queries to gate space + qnorm (no RoPE; caller applies it).

        ``[num_tokens, num_q_head, head_dim]`` (post-q_norm, pre-RoPE)
        -> ``[num_tokens, num_k_head, gate_hidden_size]``.
        """
        return self.attngate_qnorm(self.attngate_linear_q(q))

# TODO: clean up unused code (RoPE & mask)
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_seer(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply NeoX-style RoPE to a gate tensor (full-width cos/sin, rotate_half).

    Matches the reference ``apply_rotary_pos_emb_single`` (non-flash path):
    ``x_embed = x * cos + rotate_half(x) * sin``.  The gate RoPE dimension
    equals ``gate_hidden_size`` and ``cos``/``sin`` are the same width, so no
    partial-rotary handling is needed.

    Args:
        x: ``[N, H, gate_dim]``.
        cos, sin: broadcastable to ``x``, e.g. ``[N, 1, gate_dim]``.
    """
    return (x * cos) + (_rotate_half(x) * sin)


@triton.jit
def _gate_q_rope_kernel(
    q_ptr,  # [bsz, Hk, GATE_D]  gate query, projected + qnorm (any float dtype)
    q_stride_b,
    q_stride_h,
    seq_lens_ptr,  # [bsz] int32  includes the current token
    inv_freq_ptr,  # [HALF] fp32 NeoX inverse frequencies
    out_ptr,  # [bsz, Hk, GATE_D]  rotated query (may alias q_ptr -> in place)
    o_stride_b,
    o_stride_h,
    GATE_D: tl.constexpr,
    HALF: tl.constexpr,  # GATE_D // 2
    HALF_POW2: tl.constexpr,  # next_pow2(HALF)
):
    """Fuse the gate-query NeoX RoPE into one kernel (grid ``(bsz, Hk)``).

    One program rotates one ``(request, kv-head)`` gate vector at the request's
    *current* token position (``seq_len - 1``) — the same position the summary K
    vectors are RoPE'd at (see ``cache_kernels._compress_and_rope``), so scoring
    is a dot of like-rotated vectors.  The rotation is computed in fp32 (cos/sin
    from ``inv_freq``, no precomputed tables) to match the K-side numerics
    bit-for-bit; the result is cast back to the output dtype on store.

    Replaces the multi-op torch path (mul/cat/cos/sin/rotate_half/mul/add) and
    its intermediate ``cos``/``sin``/``emb`` allocations with a single launch.
    Static shapes (GATE_D fixed per config), so it stays CUDA-graph capturable.
    """
    req = tl.program_id(0)
    head = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)

    offs_h = tl.arange(0, HALF_POW2)
    h_mask = offs_h < HALF

    q_base = q_ptr + req * q_stride_b + head * q_stride_h
    q_lo = tl.load(q_base + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    q_hi = tl.load(q_base + offs_h + HALF, mask=h_mask, other=0.0).to(tl.float32)

    inv_freq = tl.load(inv_freq_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    angle = (seq_len - 1).to(tl.float32) * inv_freq
    cos_h = tl.cos(angle)
    sin_h = tl.sin(angle)
    # NeoX rotate_half: out_lo = lo*cos - hi*sin, out_hi = hi*cos + lo*sin.
    out_lo = q_lo * cos_h - q_hi * sin_h
    out_hi = q_hi * cos_h + q_lo * sin_h

    o_base = out_ptr + req * o_stride_b + head * o_stride_h
    tl.store(o_base + offs_h, out_lo.to(out_ptr.dtype.element_ty), mask=h_mask)
    tl.store(o_base + offs_h + HALF, out_hi.to(out_ptr.dtype.element_ty), mask=h_mask)


def rope_gate_query(
    q_gate: torch.Tensor,  # [bsz, Hk, gate_dim] projected + qnorm, pre-RoPE
    seq_lens: torch.Tensor,  # [bsz] includes the current token (device)
    rope_inv_freqs: torch.Tensor,  # [gate_dim//2] fp32 NeoX inverse frequencies
) -> torch.Tensor:
    """NeoX RoPE the decode gate query at each request's current position.

    Fused equivalent of building cos/sin from ``rope_inv_freqs`` at position
    ``seq_len - 1`` and calling :func:`apply_rotary_pos_emb_seer`.  Returns a new
    ``[bsz, Hk, gate_dim]`` tensor (same dtype as ``q_gate``); the rotation is
    done in fp32 internally to match the K-side summary RoPE numerics.  Static
    shapes, no host sync — CUDA-graph capturable.
    """
    bsz, Hk, gate_dim = q_gate.shape
    HALF = gate_dim // 2
    HALF_POW2 = triton.next_power_of_2(HALF)
    q_gate = q_gate.contiguous()
    out = torch.empty_like(q_gate)
    _gate_q_rope_kernel[(bsz, Hk)](
        q_gate,
        q_gate.stride(0),
        q_gate.stride(1),
        seq_lens.to(torch.int32),
        rope_inv_freqs,
        out,
        out.stride(0),
        out.stride(1),
        GATE_D=gate_dim,
        HALF=HALF,
        HALF_POW2=HALF_POW2,
    )
    return out


def select_blocks_from_scores(
    attn: torch.Tensor,
    block_valid_mask: torch.Tensor,
    sparsity_method: str,
    threshold: float,
    block_budget: int,
) -> torch.Tensor:
    """Select active blocks from softmaxed gate scores.

    Args:
        attn: ``[bsz, num_k_head, num_blocks]`` softmax block probabilities.
        block_valid_mask: ``[bsz, 1 or num_k_head, num_blocks]`` bool, True for
            blocks that exist for this request (causal/padding mask).
        sparsity_method: ``"threshold"`` or ``"token_budget"``.
        threshold: probability threshold (threshold method).
        block_budget: max number of blocks to keep (token_budget method).

    Returns:
        Bool mask ``[bsz, num_k_head, num_blocks]``; each request's last *valid*
        block is always forced on.
    """
    if sparsity_method == "token_budget":
        num_blocks = attn.size(-1)
        if num_blocks <= block_budget:
            mask = torch.ones_like(attn, dtype=torch.bool)
        else:
            _, topk_idx = torch.topk(attn, k=block_budget, dim=-1, sorted=False)
            mask = torch.zeros_like(attn, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
        mask = mask & block_valid_mask
    else:  # threshold
        mask = (attn > threshold) & block_valid_mask
    # Always attend to each request's most recent (last *valid*) block.
    #
    # NOTE: we must NOT use ``mask[:, :, -1] = True`` here.  In a batch the
    # block axis is padded to the longest request's block count, so column -1 is
    # an out-of-range (invalid) block for shorter requests; forcing it on would
    # select a block whose tokens are all beyond cache_seqlens (→ empty/NaN in
    # the sparse-decode kernel) while leaving the request's true last block
    # unselected.  Instead force the last *valid* block per (req, head).
    bsz, num_k_head, num_blocks = mask.shape
    valid = block_valid_mask.expand(bsz, num_k_head, num_blocks)
    block_ar = torch.arange(num_blocks, device=mask.device)
    last_valid = torch.where(
        valid, block_ar.view(1, 1, -1), torch.full_like(block_ar.view(1, 1, -1), -1)
    ).amax(dim=-1)  # [bsz, num_k_head], index of last valid block (>=0)
    mask.scatter_(-1, last_valid.unsqueeze(-1).clamp(min=0), True)
    return mask
