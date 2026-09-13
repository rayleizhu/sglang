"""Local compatibility shims so the vendored SeerAttention-R reference runs
without the ``flash_attn`` package (which has no prebuilt wheel for the
torch 2.9 environment here).

The reference uses ``flash_attn`` in exactly three places:

* prefill attention via ``flash_attn_varlen_func`` — replaced by a PyTorch
  ``scaled_dot_product_attention`` (mathematically equivalent causal attention;
  see ``sdpa_varlen_causal``).
* ``flash_attn.layers.rotary.apply_rotary_emb_func`` — only invoked when
  ``use_flash_rope=True``.  The released decode AttnGates use
  ``use_flash_rope=False`` (the default), so this path is never taken; we
  provide a stub that errors only if actually called.
* ``flash_attn.bert_padding`` helpers — only used by the reference's
  ``_upad_input`` unpadding path, which we bypass entirely by using the padded
  SDPA prefill above.

The decode-time Triton kernels do NOT depend on flash_attn (their
``flash_attn_with_kvcache`` import lives inside ``__main__`` benchmark code).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpa_varlen_causal(
    q: torch.Tensor,  # [B, Sq, Hq, D]
    k: torch.Tensor,  # [B, Sk, Hk, D]
    v: torch.Tensor,  # [B, Sk, Hk, D]
    attention_mask: torch.Tensor,  # [B, Sk] 1=valid (left padding assumed by ref)
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Causal multi-head attention over a padded batch, returning [B, Sq, Hq, D].

    Mathematically equivalent to the reference's
    ``flash_attn_varlen_func(..., causal=True)`` prefill, but implemented with
    PyTorch SDPA so no flash_attn dependency is needed.  GQA is handled by
    expanding KV heads.  Padding is masked out via an additive bias combined
    with a causal mask.

    Assumes prefill from position 0 (``Sq == Sk``), which is how the reference
    drives prefill (single chunk, no prefix reuse).
    """
    B, Sq, Hq, D = q.shape
    _, Sk, Hk, _ = k.shape
    assert Sq == Sk, "sdpa_varlen_causal assumes full prefill (Sq == Sk)"
    if softmax_scale is None:
        softmax_scale = D ** -0.5

    # [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Expand KV heads for GQA.
    if Hk != Hq:
        rep = Hq // Hk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    # Build a [B, 1, Sq, Sk] boolean "allow" mask: causal AND key-valid.
    device = q.device
    causal = torch.tril(
        torch.ones(Sq, Sk, dtype=torch.bool, device=device)
    )  # [Sq, Sk]
    key_valid = attention_mask.to(torch.bool)  # [B, Sk]
    allow = causal[None, None, :, :] & key_valid[:, None, None, :]
    # Guard against fully-masked rows (none here since causal keeps the diagonal).

    out = F.scaled_dot_product_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        attn_mask=allow,
        scale=softmax_scale,
    )  # [B, Hq, Sq, D]
    return out.transpose(1, 2).contiguous()  # [B, Sq, Hq, D]


def apply_rotary_emb_func(*args, **kwargs):  # pragma: no cover - guarded path
    raise RuntimeError(
        "use_flash_rope=True requires flash_attn, which is unavailable in this "
        "environment.  The released SeerAttention-R decode AttnGates use "
        "use_flash_rope=False, so this code path should never be reached."
    )
