"""SeerAttention-R latency/throughput matrix benchmark across model size, context
length, and batch size.

This is a *matrix* driver on top of the same timing methodology as
``seer_attn_speed_bench.py`` (raw random token ids via ``skip_tokenizer_init``,
one Engine per (model, backend, ctx) subprocess, prefill ≈ TTFT from a
max_new_tokens=1 generate, decode throughput from a max_new_tokens=N generate).

It adds three axes the single-shot script lacked:

  * **model size**   — 4B and 14B SeerAttention-Decode AttnGates checkpoints.
  * **batch size**   — multiple concurrent requests per (ctx) point.
  * **tensor parallel** — TP=2 across both H20s for fair, comparable numbers and
    to fit the memory-heavy cells.

IMPORTANT — memory model.  SeerAttention-R is *compute*-sparse, not *memory*-
sparse: it stores the FULL KV cache and only selects which blocks to attend.  So
KV-cache footprint is ``batch * ctx`` exactly like dense.  The worker reads the
engine's actual ``max_total_num_tokens`` (KV pool capacity) and, if
``batch * ctx`` exceeds it, records the cell as ``OOM`` instead of silently
running the requests in waves (which would produce a meaningless "throughput"
that is really sequential execution).  No silent capacity capping.

Each (model, backend, ctx) is one subprocess that sweeps the batch list; cells
that don't fit the pool are reported, not dropped.

Usage:
    python seer_attn_matrix_bench.py \
        --ckpt-4b /tmp/seer_ckpt --ckpt-14b /tmp/seer_ckpt_14b \
        --models 4b,14b --ctx-lens 8192,65536,131072 --batches 1,8,32,64 \
        --budget 4096 --decode-steps 64 --tp 2 --dense
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


def torch_sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Worker: one Engine config (model, backend, ctx); sweep the batch list.
# --------------------------------------------------------------------------
def _worker(args):
    if args.budget:
        os.environ["SGLANG_SEER_TOKEN_BUDGET"] = str(args.budget)
        os.environ.setdefault("SGLANG_SEER_START_LAYER", "0")

    # 64k/128k exceed both base models' native 40960 context.  This is purely a
    # *speed* benchmark on random tokens (output quality is irrelevant), so we
    # explicitly allow the context_length override past the derived maximum.
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    import sglang as sgl

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    max_batch = max(batches)

    # context_length must cover ctx_len + the decode steps; both base models are
    # natively 40960, so long ctx (64k/128k) needs an explicit override.  For a
    # speed benchmark on random tokens this is purely an allocation knob.
    ctx_cap = args.ctx_len + args.decode_steps + 16

    llm = sgl.Engine(
        model_path=args.model_path,
        attention_backend=args.backend,
        tp_size=args.tp,
        context_length=ctx_cap,
        mem_fraction_static=args.mem_fraction,
        max_running_requests=max(8, max_batch),
        disable_cuda_graph=args.eager,
        disable_radix_cache=True,  # no prefix reuse — measure raw prefill
        log_level="error",
        skip_tokenizer_init=True,
    )

    # KV pool capacity (tokens).  SeerAttention stores the FULL cache, so a cell
    # is feasible only if batch * ctx_len <= max_total_num_tokens.
    info = llm.get_server_info()
    max_tot = int(info.get("max_total_num_tokens") or 0)

    def _gen(input_ids, n):
        return llm.generate(
            input_ids=input_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": n},
        )

    results = []
    for batch in batches:
        need = batch * args.ctx_len
        if max_tot and need > max_tot:
            results.append(
                {
                    "ctx_len": args.ctx_len,
                    "batch": batch,
                    "status": "OOM",
                    "need_tokens": need,
                    "pool_tokens": max_tot,
                }
            )
            print(
                f"  [skip] batch={batch} ctx={args.ctx_len}: needs {need} tok > "
                f"pool {max_tot} tok",
                flush=True,
            )
            continue

        ids = _make_input_ids(args.ctx_len, batch)

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
        _gen(ids, args.decode_steps)
        torch_sync()
        t_total = time.perf_counter() - t0

        t_decode = max(t_total - t_prefill, 1e-6)
        n_decode_tok = batch * (args.decode_steps - 1)
        dec_tok_s = n_decode_tok / t_decode
        per_step_ms = 1000.0 * t_decode / max(args.decode_steps - 1, 1)

        results.append(
            {
                "ctx_len": args.ctx_len,
                "batch": batch,
                "status": "ok",
                "prefill_ms": 1000.0 * t_prefill,
                "decode_tok_s": dec_tok_s,
                "per_step_ms": per_step_ms,
                "pool_tokens": max_tot,
            }
        )
        print(
            f"  [ok]   batch={batch} ctx={args.ctx_len}: "
            f"prefill={1000.0*t_prefill:.1f}ms decode={dec_tok_s:.1f}tok/s "
            f"per_step={per_step_ms:.2f}ms",
            flush=True,
        )

    llm.shutdown()
    with open(args.out, "w") as f:
        json.dump(results, f)


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------
def _run_cell(py, model_path, backend, ctx_len, args, tag):
    out_path = f"/tmp/seer_matrix_{tag}.json"
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [
        py, __file__, "--worker",
        "--model-path", model_path,
        "--backend", backend,
        "--ctx-len", str(ctx_len),
        "--batches", args.batches,
        "--decode-steps", str(args.decode_steps),
        "--tp", str(args.tp),
        "--mem-fraction", str(args.mem_fraction),
        "--out", out_path,
    ]
    if args.budget and backend == "seer_attn":
        cmd += ["--budget", str(args.budget)]
    if args.eager:
        cmd += ["--eager"]
    print(f"\n[run] {tag}: model={os.path.basename(model_path)} "
          f"backend={backend} ctx={ctx_len} tp={args.tp}", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0 or not os.path.exists(out_path):
        print(f"  [FAIL] {tag} exited rc={rc} (likely engine OOM at init)", flush=True)
        return [{"ctx_len": ctx_len, "batch": b, "status": "init-fail"}
                for b in (int(x) for x in args.batches.split(",") if x.strip())]
    with open(out_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-4b", default="/tmp/seer_ckpt")
    ap.add_argument("--ckpt-14b", default="/tmp/seer_ckpt_14b")
    ap.add_argument("--models", default="4b,14b")
    ap.add_argument("--ctx-lens", default="8192,65536,131072")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--dense", action="store_true",
                    help="also benchmark the dense base model (flashinfer).")
    ap.add_argument("--eager", action="store_true",
                    help="force CUDA graph OFF (default graph ON, realistic serving).")
    # worker mode
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model-path")
    ap.add_argument("--backend")
    ap.add_argument("--ctx-len", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.worker:
        _worker(args)
        return

    ctx_lens = [int(c) for c in args.ctx_lens.split(",") if c.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    py = sys.executable

    ckpts = {"4b": args.ckpt_4b, "14b": args.ckpt_14b}
    base_models = {}
    for m in models:
        with open(os.path.join(ckpts[m], "config.json")) as f:
            base_models[m] = json.load(f)["base_model"]

    mode = "eager (graph OFF)" if args.eager else "graph ON"
    print(f"models={models} ctx_lens={ctx_lens} batches={args.batches} "
          f"budget={args.budget} tp={args.tp} steps={args.decode_steps} mode={mode}")

    # (model, backend, ctx) -> result list
    all_rows = []
    for m in models:
        for ctx in ctx_lens:
            # seer
            tag = f"{m}_seer{args.budget}_ctx{ctx}"
            res = _run_cell(py, ckpts[m], "seer_attn", ctx, args, tag)
            all_rows.append((m, f"seer-{args.budget}", ctx, res))
            # dense
            if args.dense:
                tag = f"{m}_dense_ctx{ctx}"
                res = _run_cell(py, base_models[m], "flashinfer", ctx, args, tag)
                all_rows.append((m, "dense", ctx, res))

    # ---- Report ----
    print("\n" + "=" * 78)
    print("SeerAttention-R matrix: latency (prefill TTFT) + decode throughput")
    print(f"(TP={args.tp}, budget={args.budget}, decode_steps={args.decode_steps}, {mode})")
    print("=" * 78)
    hdr = (f"{'model':5s} {'config':12s} {'ctx':>7s} {'batch':>5s} "
           f"{'prefill_ms':>11s} {'decode_tok/s':>13s} {'per_step_ms':>12s} {'status':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for m, cfg, ctx, res in all_rows:
        for r in res:
            st = r.get("status", "ok")
            if st == "ok":
                print(f"{m:5s} {cfg:12s} {ctx:7d} {r['batch']:5d} "
                      f"{r['prefill_ms']:11.1f} {r['decode_tok_s']:13.1f} "
                      f"{r['per_step_ms']:12.3f} {'ok':>8s}")
            else:
                detail = ""
                if st == "OOM":
                    detail = f" (need {r['need_tokens']} > pool {r['pool_tokens']} tok)"
                print(f"{m:5s} {cfg:12s} {ctx:7d} {r['batch']:5d} "
                      f"{'-':>11s} {'-':>13s} {'-':>12s} {st:>8s}{detail}")

    # ---- Speedup vs dense ----
    dense = {}
    for m, cfg, ctx, res in all_rows:
        if cfg == "dense":
            for r in res:
                if r.get("status") == "ok":
                    dense[(m, ctx, r["batch"])] = r
    if dense:
        print("\n---- seer decode throughput speedup vs dense ----")
        for m, cfg, ctx, res in all_rows:
            if cfg == "dense":
                continue
            for r in res:
                if r.get("status") != "ok":
                    continue
                d = dense.get((m, ctx, r["batch"]))
                if d:
                    sp = r["decode_tok_s"] / d["decode_tok_s"]
                    print(f"  {m:5s} {cfg:12s} ctx={ctx:6d} batch={r['batch']:3d}  {sp:.2f}x")

    # persist full matrix
    with open("/tmp/seer_matrix_all.json", "w") as f:
        json.dump([{"model": m, "config": cfg, "ctx": ctx, "results": res}
                   for m, cfg, ctx, res in all_rows], f, indent=2)
    print("\n[saved] /tmp/seer_matrix_all.json")


if __name__ == "__main__":
    main()
