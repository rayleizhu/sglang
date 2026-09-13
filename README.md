# Block-Sparse Attention for SGLang

A fork of [SGLang](https://github.com/sgl-project/sglang) (from [v0.5.9](https://github.com/rayleizhu/sglang/tree/v0.5.9)) that adds **block-sparse
decode attention** backends: prefill stays dense, and each decode step attends
only a selected subset of KV *blocks* instead of the whole context.

Two backends share one framework under
`python/sglang/srt/layers/attention/blocksparse/`:

| Backend | `--attention-backend` | Sparsity decided by | Status |
|---|---|---|---|
| SeerAttention-R | `seer_attn` | a trained per-layer AttnGate | complete, CI-tested, evaluated |
| MoBA | `moba` | mean-pooled block summaries scored against the query | experimental |

Both keep prefill dense, build per-block KV summaries during prefill, and route
each decode step to a sparse subset of blocks through the shared
`BlockSparseAttnBackend` base (KV-index expansion, FlashInfer virtual-batch
decode, CUDA-graph hooks). `seer_attn` is the more complete of the two and is
what the rest of this document focuses on.

> **This is compute-sparse, not memory-sparse.** The full KV cache is still
> stored; sparsity reduces the attention *work* per step, not the KV footprint.
> A summary pool and (for `seer_attn`) a rolling accumulator are additional
> memory on top of dense. Do not expect KV-cache savings.

---

## Quick start

```bash
python -m sglang.launch_server \
    --model-path SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates \
    --attention-backend seer_attn \
    --tp-size 1 \
    --trust-remote-code
```

A SeerAttention-R checkpoint is **two artefacts composed at load time**:

1. the **AttnGates** repo you point `--model-path` at — it contains only
   `config.json` and `attn_gate_weights.pth` (the distilled gate weights);
2. the **frozen base model** named by `config.base_model`, loaded separately.

Because the AttnGates repo ships no tokenizer files, pass the base model's
tokenizer if the checkpoint directory doesn't have one:

```python
import sglang as sgl

engine = sgl.Engine(
    model_path="SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates",
    tokenizer_path="Qwen/Qwen3-4B",
    attention_backend="seer_attn",
    trust_remote_code=True,
)
```

The backend force-disables radix/prefix cache and chunked prefill (each with a
warning), and silently disables piecewise CUDA graph / `torch.compile` — you
don't need to pass those flags yourself. **Full CUDA graph is supported and on
by default**; only decode is captured, prefill stays eager. Use
`--disable-cuda-graph` to turn it off.

---

## Configuration

`seer_attn` reads its architecture fields from the checkpoint's `config.json`
(`seerattn_gate_block_size`, `seerattn_gate_hidden_size`, and the pooling/norm
variant fields — a checkpoint that *declares* a different variant is rejected at
load). Inference knobs:

| Env var | Meaning | Default |
|---|---|---|
| `SGLANG_SEER_TOKEN_BUDGET` | tokens kept per step; `topk = budget // block_size` | checkpoint's `seerattn_token_budget`, else `4096` |
| `SGLANG_SEER_START_LAYER` | first layer that decodes sparsely (below it: dense) | checkpoint's `seerattn_start_layer`, else `0` |
| `SGLANG_SEER_ENABLE_CHUNKED_PREFILL` | opt into block-aligned chunked prefill | `0` |
| `SGLANG_SEER_BLOCK_SIZE` | block size used to align the chunked-prefill size | `64` |

A larger budget is closer to dense (less sparse); a smaller budget is more
sparse. The most recent block is always attended, so coherence is preserved
even at small budgets.

> `SGLANG_SEER_SPARSITY_METHOD` and `SGLANG_SEER_THRESHOLD` are also parsed and
> validated, but the backend implements **token-budget mode only** — setting
> `SGLANG_SEER_SPARSITY_METHOD=threshold` passes validation and is then ignored.
> Threshold sparsity is incompatible with the fixed single-plan-per-step design.

`moba` has no public checkpoint config and is driven entirely by environment
variables, all three of which are required by config validation:

```bash
export SGLANG_BSA_BLOCK_SIZE=64
export SGLANG_BSA_TOPK=16
export SGLANG_BSA_ROUTING_MODE=shared   # validated, but not currently read by the backend
```

---

## How it works

Per decode step, for each layer at or beyond `start_layer`:

1. **persist KV** and fold the new token into the block summary state;
   a block's summary is written once, on the step it fills;
2. **select blocks** — score the query against the cached per-block summaries
   and keep the top-`k` *previous* blocks, per (request, kv-head);
3. **expand to KV indices** — map selected block ids to token-level indices;
   the current (partial) block is appended unconditionally by position, so it
   is always attended;
4. **run attention** through FlashInfer using virtual batching.

Selection is top-k by score, plus — for MoBA only — a hardcoded forced "sink"
band (block 0 is always kept). There is no softmax and no threshold on the live
path. The two backends differ in how a block is *summarised*, how the scoring
query is *prepared*, and whether any blocks are force-selected.

### Key design: virtual batching

This is the central trick, and it solves a specific mismatch.

Block selection is **per-(request, kv-head)** — each kv-head picks a *different*
set of blocks, so a request no longer has one KV index list. But FlashInfer's
paged decode wrapper has no notion of per-head KV index sets.

**The fix: fold the kv-head axis into the batch axis.** Each `(request, kv_head)`
pair becomes an independent *virtual request*:

```
virt_bsz     = bsz * num_kv_heads
num_qo_heads = gqa_group_size   # the G query heads of that kv head
num_kv_heads = 1                # per virtual request
page_size    = 1
```

so it maps onto a plain `BatchDecodeWithPagedKVCacheWrapper`. The query is
reshaped `[bsz, num_q_heads, d] -> [bsz*H, G, d]`, and the KV buffer is
*reinterpreted* with a zero-copy `.view()`:

```
[pool_size, H, d]  ->  [pool_size * H, 1, d]
virtual_idx = pool_loc * H + h
```

This works because `H*d` is the contiguous trailing extent of the buffer — and
it is exactly why **`page_size == 1` is required**: the expansion gathers KV per
*logical token* through `req_to_token`, not per page.

Two consequences worth knowing:

- The sparse KV length —
  `min(current_block, topk) * block_size + current_partial_block` — depends only
  on `seq_len`, never on *which* blocks routing picked (that part is
  layer-dependent). So the plan is layer-independent and computed **once per
  step**, not per layer.
- Selected block ids are kept **ascending** with `-1` padding at the tail. The
  scatter kernel writes slot `p` at offset `p*block_size`, so the ordering is
  load-bearing for both backends.

When the context is short enough that `num_prev <= topk`, routing degenerates
to dense — and there is a CI test asserting the routing reference then produces
exactly the full dense index set, with separate parity tests pinning the Triton
kernels to that reference.

### CUDA graph

Full CUDA graph works for decode. The design constraint is that **a captured
kernel's launch configuration is frozen at capture time** — grid dims, block
dims, and buffer *addresses*. Anything that varies per step must live in the
*contents* of a fixed-address buffer, not in the shape of the launch.

That premise was originally violated by allocating a fresh `kv_indices` tensor
and rebinding the wrapper's `_paged_kv_indices_buf` on every layer. The fix:
pre-allocate one fixed buffer sized to the virtual-batch upper bound, pre-bind
it on the eager wrapper too, and have every layer write into it **in place**.
Both eager and graph paths then share a single code path.

The per-step summary kernels are fixed-grid and free of host syncs; the
"is this a fill step" and "is this a padding row" tests live *inside* the
kernels (a data-dependent branch within a kernel is fine — capture records only
the launch). Prefill is never captured, so its host-scheduled summary build is
unconstrained.

**Piecewise CUDA graph / `torch.compile` are not supported** for either backend:
the dense prefill builds block summaries via host-scheduled Triton kernels, and
for `seer_attn` the piecewise split op additionally cannot carry the per-layer
gate tensors.

---

## Limitations

All of the following are **guarded** — the server raises or force-disables; it
does not silently produce wrong results.

| Feature | Behavior with `seer_attn` |
|---|---|
| Piecewise CUDA graph / `torch.compile` | Force-disabled (prefill's summary build is host-scheduled; for `seer_attn` the split op also cannot carry the gate tensors). **Full** CUDA graph works. |
| radix / prefix cache | Force-disabled, to match the reference's single dense prefill from position 0. (It is also not reconstructible: gate summaries pool pre-RoPE keys, which the radix cache does not store.) |
| chunked prefill | Off by default. Opt in with `SGLANG_SEER_ENABLE_CHUNKED_PREFILL=1` **and** an explicit `--chunked-prefill-size`; the size is snapped down to a multiple of `SGLANG_SEER_BLOCK_SIZE` (default 64) so no block straddles two chunks. |
| Disaggregated (PD) serving | `NotImplementedError` — a decode node never runs the local dense prefill that builds the summary cache. |
| MIXED batches (`enable_mixed_chunk`) | `NotImplementedError` — decode rows would take the dense extend path. |
| Streaming sessions | `NotImplementedError` — KV-ownership transfer would leak summary slots. |
| Speculative decoding | `NotImplementedError` — the summary update assumes one new token per request per step. |
| `page_size != 1` | `NotImplementedError` — the sparse decode gathers paged KV per token. |
| Non-SeerAttention checkpoint | `ValueError` at config load. |

Additional notes:

- **Models**: Qwen3 only — `Qwen3ForCausalLM` is the only architecture routed to
  the gate-augmented model. Only the configuration every released Qwen3 gate
  uses is implemented (`Qproj` query pooling, `Kmaxminavg` key pooling, qk-norm
  and RoPE on) — a checkpoint that *declares* a different variant is rejected
  at load.
  ⚠️ A **non-Qwen3** architecture launched with `seer_attn` is normally rejected at
  config load (`ValueError`, missing `seerattn_gate_*` fields). If it somehow
  carries those fields, it is not remapped to the gate-augmented model and will
  fail at the first decode step rather than degrading gracefully.
- **Tensor parallelism is supported**: the gate's `linear_q` / `linear_k` are
  sharded along the kv-head axis. Requires `num_kv_heads` to be divisible by,
  and not smaller than, the attention-TP size (gate replication is rejected).
- **Multi-turn / agent use** works, but with radix cache off every turn re-runs a
  full dense prefill over the whole conversation — no cross-turn prefix reuse.
- **MoBA is experimental**: it has a single startup guard (`page_size`), no public
  checkpoint config, hardcoded sink/local band sizes, and no accuracy evaluation
  (its only end-to-end test is graph-vs-eager self-consistency). Do not assume
  parity with `seer_attn`.

---

## Developer Guide

Accuracy results, performance measurements, and how to run the tests and
benchmarks are documented in
[`docs/advanced_features/blocksparse_attention.md`](docs/advanced_features/blocksparse_attention.md).

---

## Acknowledgment

Built on [SGLang](https://github.com/sgl-project/sglang). The SeerAttention-R
backend implements the inference path of
[SeerAttention](https://github.com/microsoft/SeerAttention)
([arXiv:2506.08889](https://arxiv.org/abs/2506.08889)); a read-only subset of
that repository is vendored under `3rdparty/seerattention_ref/` (MIT, with the
license and scope documented there) for reference comparisons. Sparse attention
kernels are executed through [FlashInfer](https://github.com/flashinfer-ai/flashinfer).

For upstream SGLang documentation, installation, and usage, see
[docs.sglang.io](https://docs.sglang.io/).
