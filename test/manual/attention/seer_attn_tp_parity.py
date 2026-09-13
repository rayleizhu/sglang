"""TP parity check for the SeerAttention-R backend: tp=1 vs tp=2.

Verifies that sharding the AttnGate weights across attention-TP ranks
(``_shard_gate_weight`` in ``models/qwen3_seer.py``) produces the same greedy
decode as the single-GPU path.  Each config runs in its own subprocess (one
Engine at a time, fresh CUDA context) and dumps output token ids; the parent
compares them per prompt.

Greedy decoding across different TP degrees is not guaranteed bit-identical
(all-reduce changes the float reduction order, which can flip a greedy argmax
at a near-tie), exactly as documented for backend-vs-backend and batch-vs-batch
comparisons elsewhere in this suite.  So we report per-prompt first-divergence
and agreement rather than asserting strict equality, and treat high agreement
(early tokens identical, only late near-tie flips) as a pass.

Usage:
    python test/srt/seer_attn_tp_parity.py --seer-ckpt /tmp/seer_ckpt \
        --budget 8192 --max-new 64
"""

import argparse
import json
import os
import subprocess
import sys

PROMPTS = [
    "The capital of France is",
    "Count from 1 to 10:",
    "Q: What is 12 times 8? A:",
    "List the first five prime numbers:",
    "Explain why the sky is blue in one sentence.",
]
MAX_NEW = 64


def _worker(model_path, budget, tp_size, max_new, out_path):
    if budget:
        os.environ["SGLANG_SEER_TOKEN_BUDGET"] = str(budget)
    import sglang as sgl

    llm = sgl.Engine(
        model_path=model_path,
        attention_backend="seer_attn",
        tp_size=tp_size,
        mem_fraction_static=0.6,
        max_running_requests=8,
        log_level="error",
        skip_tokenizer_init=True,
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    result = []
    for p in PROMPTS:
        out = llm.generate(
            input_ids=[tok.encode(p)],
            sampling_params={"temperature": 0.0, "max_new_tokens": max_new},
        )
        result.append(out[0]["output_ids"])
    llm.shutdown()
    with open(out_path, "w") as f:
        json.dump(result, f)


def _run_config(model_path, budget, tp_size, max_new, tag, py):
    out_path = f"/tmp/seer_tp_{tag}.json"
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [
        py,
        __file__,
        "--worker",
        "--model-path",
        model_path,
        "--tp",
        str(tp_size),
        "--max-new",
        str(max_new),
        "--out",
        out_path,
    ]
    if budget:
        cmd += ["--budget", str(budget)]
    print(f"[run] {tag}: tp_size={tp_size} budget={budget} ...", flush=True)
    subprocess.run(cmd, check=True, env=dict(os.environ))
    with open(out_path) as f:
        return json.load(f)


def _first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def _agreement(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seer-ckpt", default="/tmp/seer_ckpt")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model-path")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.worker:
        _worker(args.model_path, args.budget, args.tp, args.max_new, args.out)
        return

    py = sys.executable
    tp1 = _run_config(args.seer_ckpt, args.budget, 1, args.max_new, "tp1", py)
    tp2 = _run_config(args.seer_ckpt, args.budget, 2, args.max_new, "tp2", py)

    print("\n=== tp=1 vs tp=2 token parity ===")
    all_high = True
    for i, (a, b) in enumerate(zip(tp1, tp2)):
        fd = _first_divergence(a, b)
        ag = _agreement(a, b)
        status = "IDENTICAL" if fd == -1 else f"diverge@{fd}"
        print(
            f"prompt[{i}]: {status}  agreement={ag:.3f}  "
            f"len tp1={len(a)} tp2={len(b)}"
        )
        if fd != -1 and ag < 0.9:
            all_high = False
    print(
        "\nRESULT:",
        "PASS (all prompts >=0.9 agreement / identical)"
        if all_high
        else "CHECK (a prompt diverged early - investigate)",
    )


if __name__ == "__main__":
    main()
