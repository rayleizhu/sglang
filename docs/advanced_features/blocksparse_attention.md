# Block-Sparse Attention

SGLang ships two learned/heuristic **block-sparse attention** backends that share
a common framework under `python/sglang/srt/layers/attention/blocksparse/`:

| Backend | `--attention-backend` | Sparsity decided by |
|---|---|---|
| SeerAttention-R | `seer_attn` | a trained per-layer AttnGate (this document) |
| MoBA | `moba` | mean-pooled block summaries scored against the query |

Both keep prefill dense, build per-block KV summaries during prefill, and route
each decode step to a sparse subset of blocks through the shared
`BlockSparseAttnBackend` base (KV-index expansion, flashinfer virtual-batch
decode, CUDA-graph hooks). The rest of this page documents `seer_attn`, the more
complete of the two.

---

## SeerAttention-R Decode-Sparse Attention (Qwen3)

SeerAttention-R is a **decode-time learned block-sparse attention**. Each
attention layer carries a small trained **AttnGate** that, at every decode step,
scores the historical KV *blocks* (64 tokens each) and selects only a sparse
subset for the actual attention. Prefill stays dense. This reduces decode
attention cost on long contexts while staying near-lossless on reasoning tasks.

Reference: [microsoft/SeerAttention](https://github.com/microsoft/SeerAttention)
(arXiv:2506.08889). This backend implements the **inference path** of the
released Qwen3 decode AttnGates.

> **Status:** Qwen3 only. Tensor parallelism and full CUDA graph are supported
> (piecewise CUDA graph is not). See [Limitations](#limitations).

---

## How it works

* **Prefill / extend is dense.** During prefill the backend additionally builds,
  for every *complete* 64-token block, a *gated compressed-K summary* vector
  (max/min/avg pool → gate K-projection → RMSNorm → RoPE) and seeds a per-request
  *rolling accumulator* — a running `[max | min | sum]` reduction of the trailing
  partial block's raw keys (a single small vector per request, not the whole
  block).
* **Decode is block-sparse.** Each step: fold the new key into the rolling
  `[max | min | sum]` accumulator; if a block just filled, compress it
  (`avg = sum / block_size`) into the summary cache; run the
  AttnGate (project + RoPE the query, score it against the per-block summaries,
  and take the **token-budget top-k** over previous blocks); then run a
  block-sparse flash-decode over only the selected blocks' real KV. The current
  (partial) block is appended unconditionally by position, so it is always
  attended — the selector itself forces no bands for `seer_attn`.
* Layers `< start_layer` decode densely.

The released checkpoints ship **only the gate weights** (`attn_gate_weights.pth`
+ `config.json`); the frozen base model is named in `config.base_model` and is
loaded separately.

---

## Quick start

The checkpoint to point `--model-path` at is the **AttnGates** repo (it contains
`config.json` with `base_model` + `seerattn_*` fields and `attn_gate_weights.pth`).
The base model is downloaded automatically from `config.base_model`.

```bash
python -m sglang.launch_server \
    --model-path SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates \
    --attention-backend seer_attn \
    --tp-size 1 \
    --trust-remote-code
```

The backend automatically turns the following off for exact alignment with the
reference; you do **not** need to pass them:

* radix / prefix cache (`--disable-radix-cache` is forced, with a warning)
* chunked prefill (`--chunked-prefill-size -1` is forced, with a warning)
* piecewise CUDA graph / `torch.compile` (forced off silently — the dense
  prefill's summary build is host-scheduled, and piecewise's split op also
  cannot carry the gate's extra tensors)

**Full CUDA graph is supported and on by default.** Only the decode phase is
captured (prefill stays eager); pass `--disable-cuda-graph` to turn it off.

### Offline engine

```python
import sglang as sgl

llm = sgl.Engine(
    model_path="SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates",
    attention_backend="seer_attn",
    trust_remote_code=True,
)
print(llm.generate("The capital of France is",
                   {"temperature": 0.0, "max_new_tokens": 32}))
```

If the AttnGates checkpoint has no tokenizer files, pass
`tokenizer_path="Qwen/Qwen3-4B"` (the `base_model`).

---

## Configuration

Architecture fields (block size, gate hidden size, pooling, qk-norm, RoPE) come
from the checkpoint's `config.json` and are validated at load. The **inference
knobs** below default to the reference reasoning-eval values and are overridable
via environment variables:

| Env var | Meaning | Default |
|---|---|---|
| `SGLANG_SEER_TOKEN_BUDGET` | tokens kept per step; `block_budget = budget // block_size` | checkpoint's `seerattn_token_budget`, else `4096` |
| `SGLANG_SEER_START_LAYER` | first layer that decodes sparsely (`<` runs dense) | checkpoint's `seerattn_start_layer`, else `0` |
| `SGLANG_SEER_ENABLE_CHUNKED_PREFILL` | opt into block-aligned chunked prefill | `0` |
| `SGLANG_SEER_BLOCK_SIZE` | block size used to align the chunked-prefill size | `64` |

> `SGLANG_SEER_SPARSITY_METHOD` and `SGLANG_SEER_THRESHOLD` are also parsed and
> validated, but the backend implements **token-budget mode only** — setting
> `SGLANG_SEER_SPARSITY_METHOD=threshold` passes validation and is then ignored.
> Threshold sparsity is incompatible with the fixed single-plan-per-step design.

```bash
# Tighter sparsity: keep ~2048 tokens (= 32 blocks) of KV per decode step.
SGLANG_SEER_TOKEN_BUDGET=2048 python -m sglang.launch_server \
    --model-path SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates \
    --attention-backend seer_attn
```

A larger budget → closer to dense (and slower / less sparse); a smaller budget →
more sparsity. The most recent block is always selected, so coherence is
preserved even at small budgets.

---

## Supported models

The released decode AttnGates are for Qwen3 (4B / 8B / 14B). Only the
configuration every released Qwen3 gate uses is implemented: `Qproj` query
pooling, `Kmaxminavg` key pooling, qk-norm on, RoPE on. A checkpoint declaring a
different variant is rejected at load with a clear error.

---

## Limitations

These are **guarded** (the server raises or force-disables, it will not silently
produce wrong results):

| Feature | Behavior |
|---|---|
| Piecewise CUDA graph / `torch.compile` | Force-disabled for both backends: the dense prefill builds block summaries via host-scheduled Triton kernels, and for `seer_attn` the piecewise split op additionally cannot carry the gate's `seer_*` tensors. **Full** CUDA graph works and stays enabled. |
| radix / prefix cache | Force-disabled (reference does a single dense prefill from position 0). Also not reconstructible: gate summaries pool pre-RoPE keys, which the radix cache does not store. |
| chunked prefill | Off by default. Opt in with `SGLANG_SEER_ENABLE_CHUNKED_PREFILL=1` **and** an explicit `--chunked-prefill-size`; the size is snapped down to a multiple of `SGLANG_SEER_BLOCK_SIZE` (default 64) so no block straddles two chunks. |
| Disaggregated (PD) serving | `NotImplementedError` — a decode node receives KV without running the local dense prefill that builds the summary cache and rolling accumulator. |
| MIXED batches (prefill+decode interleaved) | `NotImplementedError` (`enable_mixed_chunk`): decode rows would take the dense extend path and skip the rolling fold. |
| Streaming sessions | `NotImplementedError` (`enable_streaming_session`): KV-ownership transfer nulls `req_pool_idx`, leaking the per-block summary slots. |
| Speculative decoding | `NotImplementedError` — the rolling fold and per-block summary write assume exactly one new token per request per step. |
| `page_size != 1` | `NotImplementedError` (the sparse-decode kernel gathers paged KV per token). |
| Non-SeerAttention / non-Qproj+Kmaxminavg checkpoint | `ValueError` at config load. |

Tensor parallelism **is** supported: the AttnGate's `linear_q` / `linear_k` are
sharded along the kv-head axis per attention-TP rank. The only requirement is
that `num_kv_heads` be divisible by, and not smaller than, the attention-TP size
(gate replication is rejected).

**Multi-turn / agent (multiple tool-call rounds):** supported. With radix cache
off, each turn re-runs a full dense prefill over the whole conversation so far
(system + history + new turn), then decodes sparsely — there is no cross-turn
cache reuse to go stale. There is no prefix-cache speedup across turns, so very
long multi-turn sessions pay full prefill each turn.

**Non-Qwen3 architectures:** `Qwen3ForCausalLM` is the only architecture routed
to the gate-augmented model. A non-Qwen3 checkpoint is normally rejected at
config load (`ValueError`, missing `seerattn_gate_*` fields); one that somehow
carries those fields is not remapped and will fail at the first decode step
rather than degrading gracefully to dense.

**MoBA is experimental.** It has a single startup guard (`page_size`), no public
checkpoint config (drive it with `SGLANG_BSA_BLOCK_SIZE`, `SGLANG_BSA_TOPK` and
`SGLANG_BSA_ROUTING_MODE`, all three required by config validation — though
`routing_mode` is not currently read by the backend), hardcoded sink/local band
sizes, and no accuracy evaluation (its only end-to-end test is graph-vs-eager
self-consistency). Do not assume parity with `seer_attn`.

---

## Accuracy

AIME24, n=30, greedy, `max_new_tokens=8192`, Qwen3-4B, single H20, graded with
the reference implementation's own extraction + grading utilities:

| Config | Accuracy |
|---|---|
| dense (triton backend) | 9/30 |
| `seer_attn`, budget 2048 | 9/30 |
| `seer_attn`, budget 4096 | 10/30 |

Sparse decode reaches dense-level accuracy on this benchmark. Absolute numbers
are below the published paper because this run uses greedy decoding capped at
8192 new tokens (the paper used sampling at 32768); dense and sparse are
equally constrained here, so the comparison is apples-to-apples.

Against the official HuggingFace reference implementation on the same prompts
and grading, per-problem agreement is **26/30** at both budgets. The gate math
is a bit-exact port of the reference (diff = 0.0 with shared weights).

---

## Performance

Honest summary of what has and has not been measured:

* **At a fixed budget, the sparse attention itself is flat in context length** —
  eager per-step is 32.1 → 32.4 ms from 8K → 32K context. This is the property
  the design buys. Note that under CUDA graph the gate-scoring path is still
  O(context), so *total* per-step still grows (10.7 → 19.7 ms over the same
  range); that chain is open work, not a solved problem.
* **Full CUDA graph is worth ~3x over eager** for this path
  (31.2 → 93.1 tok/s at 8K context, Qwen3-4B / H20 / batch 1).
* **No measurement in this repository shows block-sparse beating dense
  end-to-end.** The only comparison with both sides eager has sparse decode at
  **0.35x dense** (~3x slower). A graph-on-both-sides comparison against dense
  has not been run, so no speedup claim is made.

Profiling shows the residual cost is **not in the attention computation**: GEMM
(gate projections + the virtual-batch call) is ~52% of the decode step, while
block-selection routing is only ~16%. The per-step auxiliary kernels and fixed
reshape overhead dominate. Optimizing these is open work.

When quoting throughput numbers, always state the CUDA-graph state of *both*
sides — mixing graph-replayed dense against eager sparse conflates sparsity with
graph launch-overhead savings.

Remember that this is **compute-sparse, not memory-sparse**: the full KV cache is
still stored, and the summary pool plus (for `seer_attn`) the rolling accumulator
are additional memory on top of dense.

---

## Verification

Self-contained kernel/gate unit tests (no checkpoint / no network) — 35 cases
across the three registered CI files:

```bash
python3 -m pytest test/registered/attention/test_blocksparse_routing.py \
                  test/registered/attention/test_blocksparse_summary_kernel.py \
                  test/registered/attention/test_seer_attn_backend.py -q
```

These cover exact integer-index routing properties, Triton-vs-PyTorch kernel
parity, gate shapes, and TP weight sharding.

Checkpoint-based scripts under `test/manual/attention/` (need the AttnGates
checkpoint, multiple GPUs, or long runtimes — 8 scripts in total):

* `seer_attn_sparse_vs_dense.py` — sparse vs dense baseline, per-prompt bsz=1.
* `seer_attn_cudagraph_check.py` — full CUDA graph vs eager, token-identity +
  padded-batch (no-NaN) replay check.
* `moba_cudagraph_check.py` — the same graph-vs-eager check for `moba`.
* `seer_attn_tp_parity.py` — tp=1 vs tp=2 greedy parity.
* `seer_attn_vs_hf_reference.py` — vs the vendored microsoft reference oracle
  (`3rdparty/seerattention_ref/`); defaults to a checked-in golden fixture.
  Note that the vendored oracle's prefill runs through PyTorch SDPA rather than
  `flash_attn` (no prebuilt wheel for this torch version), so it is
  mathematically equivalent to the upstream reference but not bit-identical.
* `seer_attn_prefill_split_parity.py` — the three-stage prefill summary build vs
  a pure-PyTorch reference.
* `seer_attn_radix_parity.py` — radix-select vs sort-based threshold kernel.
* `seer_attn_eval_aime.py` — AIME24 accuracy (dense vs seer budgets).

Performance benchmarks live under `benchmark/blocksparse_attn/` (5 scripts:
`seer_attn_speed_bench.py`, `seer_attn_matrix_bench.py`, `seer_attn_profile.py`,
and the two block-selection micro-benchmarks).

> **Comparing attention backends:** always compare with a fixed batch
> (per-prompt `bsz=1`). Every backend (dense Triton included) is numerically
> batch-dependent — the reduction order changes with batch composition and can
> flip a greedy argmax — so batched comparisons produce spurious "mismatches".
