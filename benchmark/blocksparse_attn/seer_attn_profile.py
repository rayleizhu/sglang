"""SeerAttention-R (block-sparse) per-kernel decode profiler.

Companion to ``seer_attn_speed_bench.py`` (end-to-end tok/s) and
``seer_attn_sparse_vs_dense.py`` (correctness).  Where the speed bench tells you
*how fast* a step is, this script tells you *where the time goes* — it captures a
``torch.profiler`` chrome trace inside the scheduler subprocess and aggregates
**GPU kernel time by kernel name** over the decode window, so a kernel change can
be judged on the actual bottleneck rather than wall-clock alone.

Why eager (CUDA graph OFF): under graph replay every per-step kernel collapses
into a single opaque ``cudaGraphLaunch``, so the trace shows one blob and no
per-kernel attribution is possible.  Eager keeps every launch named, which is
exactly what we want while deciding which kernel to optimize.  (Graph-on
throughput is the speed bench's job.)

Method (one Engine, ``disable_radix_cache=True`` so prefill is never reused —
matches the speed bench):
  * warmup generate (compile / autotune Triton kernels)
  * ``start_profile(num_steps=decode_steps, activities=["CPU","GPU"])``
  * one ``generate(max_new_tokens=decode_steps)`` — the trace then holds 1
    prefill forward + (decode_steps-1) decode forwards.
  * ``stop_profile`` → chrome ``*.trace.json.gz`` in the output dir.

The trace is then parsed: GPU kernels (``cat=="kernel"``) are grouped by a
*normalized* name (Triton autotune suffixes / template args stripped) and summed.
Kernels that fire on the decode loop appear ``~(decode_steps-1)*num_layers``
times; prefill-only kernels appear ``~1`` time, so the report splits "decode
loop" (high count) from "prefill / one-shot" (low count) and reports a per-step
breakdown for the decode-loop kernels — the real optimization target list.

Usage:
    .pixi/envs/default/bin/python test/srt/seer_attn_profile.py \
        --seer-ckpt /tmp/seer_ckpt --ctx-len 32768 --budget 2048 \
        --decode-steps 64
    # dense baseline for comparison:
    .pixi/envs/default/bin/python test/srt/seer_attn_profile.py \
        --seer-ckpt /tmp/seer_ckpt --ctx-len 32768 --backend flashinfer \
        --decode-steps 64
"""

import argparse
import glob
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict

_TOK_LO, _TOK_HI = 1000, 30000


def _make_input_ids(ctx_len, batch, seed=0):
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
# Kernel-name normalization: collapse autotune / template noise so the same
# logical kernel aggregates into one row.
# --------------------------------------------------------------------------
def _normalize(name):
    # Triton kernels are emitted as e.g. "triton_poi_fused_..._0d1d2d" or the
    # python fn name with a hash suffix; flashinfer / cutlass kernels carry long
    # template args in <...>.  Strip <...>, trailing hashes, and arg lists.
    n = re.sub(r"<.*?>", "", name)
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"\bvoid\b", "", n)
    # collapse repeated whitespace
    n = re.sub(r"\s+", " ", n).strip()
    # Triton: keep the human-readable kernel stem (drop trailing _<digits> tag)
    m = re.match(r"^(triton_)?([A-Za-z_][A-Za-z0-9_]*?)(_[0-9a-f]{2,})?$", n)
    if m and m.group(2):
        return m.group(2)
    return n


# Map normalized kernel stems to a coarse "stage" bucket so the report rolls up
# routing vs attention vs cache-maintenance vs everything-else.
def _bucket(norm):
    s = norm.lower()
    if "score_blocks" in s or "select_from_scores" in s or "sort" in s:
        return "routing/select"
    if "rope_gate" in s or "gate" in s:
        return "routing/gate-rope"
    if "scatter_kv" in s or "kv_inds" in s or "block_inds" in s:
        return "routing/kv-index"
    if "summary" in s or "compress" in s or "rolling" in s or "seed" in s:
        return "cache/summary"
    if (
        "batchdecode" in s
        or "batchprefill" in s
        or "decode_kernel" in s
        or "attention" in s
        or "flashinfer" in s
        or "paged" in s
        or "single_decode" in s
    ):
        return "attention"
    if (
        "gemm" in s
        or "matmul" in s
        or "cutlass" in s
        or "linear" in s
        or "cublas" in s
        or "nvjet" in s  # NVIDIA cuBLASLt GEMM kernels (qkv/o proj, FFN)
        or "splitkreduce" in s
        or "ampere_" in s
        or "sm80_" in s
        or "sm90_" in s
    ):
        return "gemm/proj"
    if "rmsnorm" in s or "norm" in s or "layernorm" in s:
        return "norm"
    if "rotary" in s or "fused_rope" in s:
        return "rope(qkv)"
    if "act_and_mul" in s or "silu" in s or "elementwise" in s or "store_kvcache" in s:
        return "elementwise/act"
    if "embedding" in s or "embed" in s:
        return "embed"
    return "other"


def _parse_trace(path, decode_steps, prefill_cut_us=1000.0):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", [])

    # PROBLEM: flashinfer emits the SAME kernel name
    # (``BatchPrefillWithPagedKVCache``) for the one-shot dense prefill forward
    # AND for the per-step sparse virtual-batch decode.  A single
    # generate(max_new_tokens=N) = 1 prefill forward + (N-1) decode forwards, so
    # name-only aggregation buries the ~tens-of-ms/layer prefill calls inside the
    # cheap decode row and the per-step numbers explode.
    #
    # FIX: split by TIME.  The (single) prefill forward runs first and its
    # attention/GEMM kernels are huge (>>10ms each at long ctx); every decode
    # kernel is well under that.  So t_cut = the latest end-time of any kernel
    # longer than ``prefill_cut_us`` marks the prefill→decode boundary; aggregate
    # only kernels that START at/after t_cut.  If no kernel is that long (e.g.
    # dense baseline at short ctx), t_cut stays 0 and everything is counted.
    kernel_evs = [
        e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"
    ]
    t_cut = 0.0
    for e in kernel_evs:
        dur = float(e.get("dur", 0.0))
        if dur >= prefill_cut_us:
            t_cut = max(t_cut, float(e.get("ts", 0.0)) + dur)

    agg = defaultdict(lambda: {"dur": 0.0, "count": 0})
    total_gpu = 0.0
    n_excluded = 0
    for e in kernel_evs:
        if float(e.get("ts", 0.0)) < t_cut:
            n_excluded += 1
            continue
        dur = float(e.get("dur", 0.0))  # microseconds
        norm = _normalize(e.get("name", "?"))
        agg[norm]["dur"] += dur
        agg[norm]["count"] += 1
        total_gpu += dur
    if t_cut > 0:
        print(
            f"[profile] prefill cut at ts={t_cut:.0f}us; excluded {n_excluded} "
            f"pre-decode kernel events (prefill forward).",
            flush=True,
        )
    return agg, total_gpu


def _report(agg, total_gpu, decode_steps, tag):
    # Prefill is already excluded by time-window in _parse_trace, so everything
    # here is decode-window GPU time.  loop_thresh only separates per-step loop
    # kernels (count ≈ (decode_steps-1) for per-forward batch ops, or
    # ≈ (decode_steps-1)*num_layers for per-layer ops) from rare one-off setup
    # kernels; a fraction of the decode-step count is the right cut.
    loop_thresh = max(2, (decode_steps - 1) // 2)

    rows = []
    for norm, v in agg.items():
        rows.append((norm, v["dur"], v["count"], _bucket(norm)))
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"\n================ kernel profile: {tag} ================")
    print(f"total GPU kernel time in window: {total_gpu/1000:.2f} ms")
    print(f"(decode_steps={decode_steps}; loop-kernel count threshold={loop_thresh})")
    print(
        f"\n{'kernel':40s} {'bucket':18s} {'tot_ms':>9s} "
        f"{'count':>7s} {'us/call':>9s} {'%gpu':>6s}"
    )
    print("-" * 95)
    for norm, dur, count, bucket in rows:
        if dur / max(total_gpu, 1e-9) < 0.001 and count < loop_thresh:
            continue
        print(
            f"{norm[:40]:40s} {bucket:18s} {dur/1000:9.3f} "
            f"{count:7d} {dur/max(count,1):9.2f} {100*dur/max(total_gpu,1e-9):6.1f}"
        )

    # ---- Stage rollup ----
    stage = defaultdict(lambda: {"dur": 0.0, "count": 0})
    for norm, dur, count, bucket in rows:
        stage[bucket]["dur"] += dur
        stage[bucket]["count"] += count
    print(f"\n---- stage rollup ({tag}) ----")
    print(f"{'stage':20s} {'tot_ms':>9s} {'%gpu':>6s}")
    print("-" * 40)
    for b, v in sorted(stage.items(), key=lambda kv: kv[1]["dur"], reverse=True):
        print(f"{b:20s} {v['dur']/1000:9.3f} {100*v['dur']/max(total_gpu,1e-9):6.1f}")

    # ---- Per-decode-step breakdown (decode-loop kernels only) ----
    loop_total = sum(d for _, d, c, _ in rows if c >= loop_thresh)
    n_steps = max(decode_steps - 1, 1)
    print(f"\n---- decode-loop kernels, per-step ({tag}) ----")
    print(f"loop GPU time {loop_total/1000:.2f} ms over {n_steps} steps "
          f"= {loop_total/1000/n_steps:.3f} ms/step (GPU-busy; excludes gaps)")
    loop_stage = defaultdict(float)
    for norm, dur, count, bucket in rows:
        if count >= loop_thresh:
            loop_stage[bucket] += dur
    print(f"{'stage':20s} {'ms/step':>9s} {'%loop':>6s}")
    print("-" * 40)
    for b, d in sorted(loop_stage.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{b:20s} {d/1000/n_steps:9.4f} {100*d/max(loop_total,1e-9):6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seer-ckpt", default="/tmp/seer_ckpt")
    ap.add_argument("--ctx-len", type=int, default=32768)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument(
        "--backend",
        default="seer_attn",
        help="seer_attn (profile the block-sparse path) or flashinfer (dense baseline)",
    )
    ap.add_argument("--out-dir", default="/tmp/seer_prof")
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="skip the engine run; re-parse the newest trace already in --out-dir "
        "(use after editing the parser to avoid relaunching).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    is_seer = args.backend == "seer_attn"
    tag = f"{args.backend}" + (f"-{args.budget}" if is_seer else "") + f"@ctx{args.ctx_len}"

    if args.parse_only:
        traces = sorted(
            glob.glob(os.path.join(args.out_dir, "*.trace.json*")),
            key=os.path.getmtime,
        )
        if not traces:
            print(f"!! no trace found in {args.out_dir}", file=sys.stderr)
            sys.exit(1)
        trace_path = traces[-1]
        print(f"[profile] (parse-only) parsing trace: {trace_path}", flush=True)
        agg, total_gpu = _parse_trace(trace_path, args.decode_steps)
        _report(agg, total_gpu, args.decode_steps, tag)
        return

    # clear old traces so we pick up ours
    for p in glob.glob(os.path.join(args.out_dir, "*.trace.json*")):
        os.remove(p)

    is_seer = args.backend == "seer_attn"
    if is_seer:
        os.environ["SGLANG_SEER_TOKEN_BUDGET"] = str(args.budget)
        os.environ.setdefault("SGLANG_SEER_START_LAYER", "0")
    os.environ["SGLANG_TORCH_PROFILER_DIR"] = args.out_dir

    if is_seer:
        with open(os.path.join(args.seer_ckpt, "config.json")) as f:
            model_path = args.seer_ckpt
    else:
        with open(os.path.join(args.seer_ckpt, "config.json")) as f:
            model_path = json.load(f)["base_model"]

    import sglang as sgl
    import torch

    tag = f"{args.backend}" + (f"-{args.budget}" if is_seer else "") + f"@ctx{args.ctx_len}"
    print(f"[profile] {tag}  model={model_path}", flush=True)

    llm = sgl.Engine(
        model_path=model_path,
        attention_backend=args.backend,
        mem_fraction_static=0.7,
        max_running_requests=max(8, args.batch),
        disable_cuda_graph=True,  # EAGER — keep per-kernel names visible
        disable_radix_cache=True,
        log_level="error",
        skip_tokenizer_init=True,
    )

    ids = _make_input_ids(args.ctx_len, args.batch)

    def _gen(n):
        return llm.generate(
            input_ids=ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": n},
        )

    # Warmup: compile/autotune Triton + flashinfer plans, allocate buffers.
    _gen(4)
    torch.cuda.synchronize()

    llm.start_profile(
        output_dir=args.out_dir,
        num_steps=args.decode_steps,
        activities=["CPU", "GPU"],
        with_stack=False,
        record_shapes=False,
    )
    _gen(args.decode_steps)
    torch.cuda.synchronize()
    try:
        llm.stop_profile()
    except Exception:
        pass  # num_steps may auto-stop; stop_profile then no-ops

    # give the subprocess a moment to flush the trace to disk
    time.sleep(2.0)
    llm.shutdown()

    traces = sorted(
        glob.glob(os.path.join(args.out_dir, "*.trace.json*")),
        key=os.path.getmtime,
    )
    if not traces:
        print(f"!! no trace found in {args.out_dir}", file=sys.stderr)
        sys.exit(1)
    trace_path = traces[-1]
    print(f"[profile] parsing trace: {trace_path}", flush=True)
    agg, total_gpu = _parse_trace(trace_path, args.decode_steps)
    _report(agg, total_gpu, args.decode_steps, tag)


if __name__ == "__main__":
    main()
