# Vendored SeerAttention-R reference (test oracle)

This directory is a **vendored, read-only subset** of Microsoft's
SeerAttention repository (https://github.com/microsoft/SeerAttention,
arXiv:2506.08889), used **only by tests** as a reference oracle for validating
SGLang's `seer_attn` attention backend.  It is not imported by the SGLang
runtime.

Upstream license: MIT (see `LICENSE`, copied verbatim from the source repo).

## What was vendored

The decode-sparse inference path needed to instantiate
`SeerDecodingQwen3ForCausalLM` and run `batch_exist_generate`:

- `decode_sparse/` — Qwen3 modeling + config, AttnGate inference, K-compression
  cache, dense/sparse attention forwards
- `modules/` — RMSNorm, RoPE helpers (`common.py`), layernorm
- `kernels/varlen/` — Triton block-sparse / flash decode kernels, oracle sparse
- `utils.py` — model output dataclasses

## Local modifications (to run without flash_attn)

The reference depends on `flash_attn` only for the **prefill** attention
(`flash_attn_varlen_func`); the decode Triton kernels do not.  Since there is no
prebuilt `flash_attn` wheel for this environment's torch 2.9 (the torch 2.8
wheel is ABI-incompatible), the following minimal, behavior-preserving edits
were made:

1. **`flash_compat.py`** (new) — `sdpa_varlen_causal` replaces the prefill
   `flash_attn_varlen_func` with a mathematically-equivalent PyTorch SDPA
   causal attention; `apply_rotary_emb_func` is a guard stub (only reachable
   when `use_flash_rope=True`, which the released checkpoints do not use).
2. `attention_forward_dense.py` / `attention_forward_sparse.py` — prefill branch
   switched to `sdpa_varlen_causal`.
3. `attn_gate_inf.py`, `modeling_qwen3_seerattn_inference.py`, `oracle_sparse.py`
   — `from seer_attn.*` absolute imports rewritten to package-relative; flash
   RoPE import routed through `flash_compat`.
4. `modules/common.py` — `flash_attn.bert_padding` import made lazy (only used
   by the now-bypassed unpad path).
5. `cache_utils.py` — `KCompressionCache` no longer subclasses
   `transformers.cache_utils.Cache` (newer transformers changed `Cache.__init__`
   to require a `layers` argument; the class is just a per-layer container).

The decode Triton kernels, gate math, cache logic, and modeling code are
otherwise unchanged.  See `docs/advanced_features/blocksparse_attention.md` for
the alignment story and the prefill-SDPA caveat.
