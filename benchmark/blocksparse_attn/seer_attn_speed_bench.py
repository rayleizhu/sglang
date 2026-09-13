"""SeerAttention-R (block-sparse) speed benchmark — run after every kernel change.

Companion to ``seer_attn_sparse_vs_dense.py`` (which checks *correctness*).  This
script measures *generation speed* of the block-sparse backend so that a kernel
change can be judged on both axes:

  * **prefill latency** (ms)  — exercises ``update_summary_cache_prefill`` +
    dense paged prefill, at a controllable context length.
  * **decode throughput** (tok/s, and per-step ms) — exercises the per-step
    decode kernels under optimisation: rolling-accumulator fold + gated summary
    compression (``update_summary_cache_decode``), gate scoring + top-k
    (gate-query RoPE + the shared ``compute_active_block_ids``), and the
    kv-index scatter (``_scatter_kv_inds_kernel``), plus the flashinfer
    virtual-batch sparse decode itself.

The benchmark uses *raw random token ids* (``skip_tokenizer_init``) so the
context length is exactly controllable and content-independent — speed does not
depend on what the tokens say.  CUDA graph is disabled for these backends, so
per-step decode latency directly reflects kernel cost (no graph replay hiding
launch overhead) — which is exactly what we want while iterating on kernels.

Each config runs in its own subprocess (one Engine at a time).  Timing method
(per config, per (ctx_len, batch) point):
  * warmup generate, then
  * t_prefill  ≈ wall-clock of a max_new_tokens=1 generate (TTFT)
  * t_total    = wall-clock of a max_new_tokens=N generate
  * decode tok/s = batch * (N - 1) / (t_total - t_prefill)

Configs:
  * seer-<budget>  — seer_attn backend at the given token budget(s)
  * dense          — base model + flashinfer backend (upper-bound on quality,
                     lower-bound on speed at long context); optional via --dense.

Usage:
    python seer_attn_speed_bench.py --seer-ckpt /tmp/seer_ckpt \
        --ctx-lens 4096,16384 --budgets 2048,4096 \
        --decode-steps 128 --batch 1 --dense
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Random-token range kept away from special/low ids; content is irrelevant to
# speed, only the *count* of tokens matters.
_TOK_LO, _TOK_HI = 1000, 30000


def _make_input_ids(ctx_len, batch, seed=0):
    # Deterministic pseudo-random ids without importing numpy/torch in the
    # driver; a simple LCG is plenty for "arbitrary but fixed" filler tokens.
    state = seed * 2654435761 + 12345
    out = []
    for _ in range(batch):
        row = []
        for _ in range(ctx_len):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            row.append(_TOK_LO + state % (_TOK_HI - _TOK_LO))
        out.append(row)
    return out


# --------------------------------------------------------------------------
# Worker: one Engine config, time prefill + decode across (ctx_len, batch).
# --------------------------------------------------------------------------
def _worker(model_path, backend, budget, ctx_lens, batch, decode_steps, disable_graph, out_path):
    if budget:
        os.environ["SGLANG_SEER_TOKEN_BUDGET"] = str(budget)
        os.environ.setdefault("SGLANG_SEER_START_LAYER", "0")

    import sglang as sgl

    # CUDA graph: seer_attn now supports full decode graph, as does the dense
    # flashinfer baseline.  By default we measure BOTH with graph ON — that is
    # the realistic serving configuration and a fair apples-to-apples decode
    # comparison.  Pass --eager to force both eager (isolates raw per-step kernel
    # cost / measures the graph speedup itself).
    llm = sgl.Engine(
        model_path=model_path,
        attention_backend=backend,
        mem_fraction_static=0.7,
        max_running_requests=max(8, batch),
        disable_cuda_graph=disable_graph,
        disable_radix_cache=True,  # no prefix reuse — measure raw prefill
        log_level="error",
        skip_tokenizer_init=True,
    )

    def _gen(input_ids, n):
        return llm.generate(
            input_ids=input_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": n},
        )

    results = []
    for ctx_len in ctx_lens:
        ids = _make_input_ids(ctx_len, batch)

        # Warmup (compile / autotune Triton kernels, allocate buffers).
        _gen(ids, 4)
        torch_sync()

        # Prefill latency ≈ TTFT (max_new_tokens=1).
        t0 = time.perf_counter()
        _gen(ids, 1)
        torch_sync()
        t_prefill = time.perf_counter() - t0

        # Full decode run.
        t0 = time.perf_counter()
        _gen(ids, decode_steps)
        torch_sync()
        t_total = time.perf_counter() - t0

        t_decode = max(t_total - t_prefill, 1e-6)
        n_decode_tok = batch * (decode_steps - 1)
        dec_tok_s = n_decode_tok / t_decode
        per_step_ms = 1000.0 * t_decode / max(decode_steps - 1, 1)

        results.append(
            {
                "ctx_len": ctx_len,
                "batch": batch,
                "prefill_ms": 1000.0 * t_prefill,
                "decode_tok_s": dec_tok_s,
                "per_step_ms": per_step_ms,
            }
        )

    llm.shutdown()
    with open(out_path, "w") as f:
        json.dump(results, f)


def torch_sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------
def _run_config(model_path, backend, budget, ctx_lens, batch, steps, tag, py, eager):
    out_path = f"/tmp/seer_speed_{tag}.json"
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [
        py, __file__, "--worker",
        "--model-path", model_path,
        "--backend", backend,
        "--ctx-lens", ",".join(str(c) for c in ctx_lens),
        "--batch", str(batch),
        "--decode-steps", str(steps),
        "--out", out_path,
    ]
    if budget:
        cmd += ["--budget", str(budget)]
    if eager:
        cmd += ["--eager"]
    mode = "eager" if eager else "graph"
    print(f"[run] {tag}: backend={backend} budget={budget} mode={mode} ...", flush=True)
    subprocess.run(cmd, check=True)
    with open(out_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seer-ckpt", default="/tmp/seer_ckpt")
    ap.add_argument("--ctx-lens", default="4096,16384")
    ap.add_argument("--budgets", default="2048,4096")
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument(
        "--dense",
        action="store_true",
        help="also benchmark the dense base model (flashinfer) for comparison.",
    )
    ap.add_argument(
        "--eager",
        action="store_true",
        help="force CUDA graph OFF for every config (both seer and dense). "
        "Default runs graph ON (realistic serving). Run once each way to read "
        "off the graph speedup directly.",
    )
    # worker mode
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model-path")
    ap.add_argument("--backend")
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    ctx_lens = [int(c) for c in args.ctx_lens.split(",") if c.strip()]

    if args.worker:
        _worker(
            args.model_path,
            args.backend,
            args.budget or None,
            ctx_lens,
            args.batch,
            args.decode_steps,
            args.eager,  # disable_graph
            args.out,
        )
        return

    with open(os.path.join(args.seer_ckpt, "config.json")) as f:
        base_model = json.load(f)["base_model"]
    print(f"base_model = {base_model}")
    mode = "eager (graph OFF)" if args.eager else "graph ON"
    print(
        f"ctx_lens={ctx_lens} batch={args.batch} decode_steps={args.decode_steps} "
        f"mode={mode}",
        flush=True,
    )

    py = sys.executable
    rows = []  # (tag, result_list)

    if args.dense:
        rows.append(
            (
                "dense",
                _run_config(
                    base_model, "flashinfer", 0, ctx_lens, args.batch,
                    args.decode_steps, "dense", py, args.eager,
                ),
            )
        )
    for b in args.budgets.split(","):
        b = b.strip()
        if not b:
            continue
        tag = f"seer-{b}"
        rows.append(
            (
                tag,
                _run_config(
                    args.seer_ckpt, "seer_attn", int(b), ctx_lens, args.batch,
                    args.decode_steps, tag.replace("-", ""), py, args.eager,
                ),
            )
        )

    # ---- Report ----
    print("\n================ SeerAttention speed ================")
    print(
        f"{'config':12s} {'ctx':>7s} {'prefill_ms':>11s} "
        f"{'decode_tok/s':>13s} {'per_step_ms':>12s}"
    )
    print("-" * 60)
    for tag, res in rows:
        for r in res:
            print(
                f"{tag:12s} {r['ctx_len']:7d} {r['prefill_ms']:11.1f} "
                f"{r['decode_tok_s']:13.1f} {r['per_step_ms']:12.3f}"
            )

    # ---- Speedup vs dense, if available ----
    dense_res = dict(
        ((r["ctx_len"], r["batch"]), r)
        for tag, res in rows
        if tag == "dense"
        for r in res
    )
    if dense_res:
        print("\n---- decode speedup vs dense ----")
        for tag, res in rows:
            if tag == "dense":
                continue
            for r in res:
                d = dense_res.get((r["ctx_len"], r["batch"]))
                if d:
                    sp = r["decode_tok_s"] / d["decode_tok_s"]
                    print(f"  {tag:12s} ctx={r['ctx_len']:6d}  {sp:.2f}x")


if __name__ == "__main__":
    main()
