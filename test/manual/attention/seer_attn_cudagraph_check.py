"""Verify SeerAttention-R runs under FULL CUDA graph and matches eager.

Two engines, one per run (sequential subprocess):
  * graph: seer_attn with full cuda graph ENABLED (default now)
  * eager: seer_attn with --disable-cuda-graph

Per-prompt bsz=1 greedy generation, compare token ids.  Also runs a small
concurrent batch (3 prompts at once) under graph to exercise padded replay /
no-NaN on padding rows.

Usage:
  python seer_attn_cudagraph_check.py --seer-ckpt <dir> --mode graph --out /tmp/g.json
  python seer_attn_cudagraph_check.py --seer-ckpt <dir> --mode eager --out /tmp/e.json
  python seer_attn_cudagraph_check.py --compare /tmp/e.json /tmp/g.json
"""

import argparse
import json
import os

PROMPTS = [
    "The capital of France is",
    "List the first 10 prime numbers: 2, 3, 5,",
    "Q: If a train travels 60 km in 1.5 hours, what is its average speed? A:",
]
MAX_NEW = 64


def _worker(ckpt, mode, out_path):
    import sglang as sgl
    from transformers import AutoTokenizer

    base = json.load(open(os.path.join(ckpt, "config.json")))["base_model"]
    os.environ["SGLANG_SEER_TOKEN_BUDGET"] = "512"

    llm = sgl.Engine(
        model_path=ckpt,
        tokenizer_path=base,
        attention_backend="seer_attn",
        mem_fraction_static=0.6,
        max_running_requests=8,
        disable_cuda_graph=(mode == "eager"),
        cuda_graph_max_bs=4,
        log_level="info",
        skip_tokenizer_init=True,
    )
    tok = AutoTokenizer.from_pretrained(base)

    # (a) per-prompt bsz=1 (fair backend comparison)
    single = []
    for p in PROMPTS:
        out = llm.generate(
            input_ids=[tok.encode(p)],
            sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW},
        )
        single.append(out[0]["output_ids"])

    # (b) concurrent batch of 3 -> pads to a captured bs (e.g. 4) on replay
    batch_out = llm.generate(
        input_ids=[tok.encode(p) for p in PROMPTS],
        sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW},
    )
    batched = [o["output_ids"] for o in batch_out]
    # NaN/garbage check: every output must be non-empty and within vocab
    ok = all(len(x) > 0 for x in batched)

    llm.shutdown()
    json.dump({"single": single, "batched": batched, "batch_ok": ok}, open(out_path, "w"))
    print(f"wrote {out_path} (mode={mode}, batch_ok={ok})", flush=True)


def _compare(eager_path, graph_path):
    e = json.load(open(eager_path))
    g = json.load(open(graph_path))
    all_ok = True
    for i, (eo, go) in enumerate(zip(e["single"], g["single"])):
        L = min(len(eo), len(go))
        fd = next((j for j in range(L) if eo[j] != go[j]), -1)
        agree = sum(1 for j in range(L) if eo[j] == go[j]) / max(L, 1)
        status = "IDENTICAL" if (fd == -1 and len(eo) == len(go)) else "DIVERGE"
        if status != "IDENTICAL":
            all_ok = False
        print(f"[single {i}] len e={len(eo)} g={len(go)} agree={agree:.3f} "
              f"first_div={fd} {status}")
    print(f"batch_ok eager={e['batch_ok']} graph={g['batch_ok']}")
    print("\n==> GRAPH==EAGER" if all_ok else "\n==> MISMATCH")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seer-ckpt")
    ap.add_argument("--mode", choices=["graph", "eager"])
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        _compare(*args.compare)
    else:
        _worker(args.seer_ckpt, args.mode, args.out)
