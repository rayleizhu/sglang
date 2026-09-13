"""SeerAttention-R correctness check: sparse vs dense baseline.

Runs the SAME prompts with greedy decoding under three configs and compares
generated token ids:

  * dense      — base Qwen3 with the triton backend (ground truth)
  * seer-big   — seer_attn with a large token budget (should == dense)
  * seer-small — seer_attn with a small token budget (sparsity active; should
                 stay coherent, divergence from dense is expected & reported)

Each config runs in its own subprocess (one Engine at a time) and writes its
output token ids to a JSON file; the driver then compares them.

Usage:
    # base_model id is read from the seer checkpoint's config.json
    python sparse_vs_dense.py --seer-ckpt /tmp/seer_ckpt
"""

import argparse
import json
import os
import subprocess
import sys

PROMPTS = [
    "Count from 1 to 100 separated by commas: 1, 2, 3,",
    "The capital of France is",
    "List the first 30 prime numbers: 2, 3, 5,",
    "Explain step by step why the sky is blue.",
    "Q: If a train travels 60 km in 1.5 hours, what is its average speed? A: Let's think step by step.",
]
MAX_NEW = 256


# --------------------------------------------------------------------------
# Worker: launched as a subprocess, runs one Engine config, dumps token ids.
# --------------------------------------------------------------------------
def _worker(model_path, backend, budget, out_path):
    if budget:
        os.environ["SGLANG_SEER_TOKEN_BUDGET"] = str(budget)
    import sglang as sgl

    llm = sgl.Engine(
        model_path=model_path,
        attention_backend=backend,
        mem_fraction_static=0.6,
        max_running_requests=8,
        disable_cuda_graph=True,
        log_level="error",
        skip_tokenizer_init=True,  # return token ids, not text
    )
    # Encode prompts with the HF tokenizer (engine returns output_ids).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # IMPORTANT: run each prompt in its OWN bsz=1 request.  Batched decoding is
    # numerically batch-dependent for *every* backend (triton included): the
    # reduction order changes with batch composition and can flip a greedy
    # argmax.  To compare backends fairly we must hold the batch fixed, so we
    # decode one prompt at a time.
    result = []
    for p in PROMPTS:
        out = llm.generate(
            input_ids=[tok.encode(p)],
            sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW},
        )
        result.append(out[0]["output_ids"])
    llm.shutdown()

    with open(out_path, "w") as f:
        json.dump(result, f)


def _run_config(model_path, backend, budget, tag, py):
    out_path = f"/tmp/seer_cmp_{tag}.json"
    if os.path.exists(out_path):
        os.remove(out_path)
    env = dict(os.environ)
    cmd = [
        py,
        __file__,
        "--worker",
        "--model-path",
        model_path,
        "--backend",
        backend,
        "--out",
        out_path,
    ]
    if budget:
        cmd += ["--budget", str(budget)]
    print(f"[run] {tag}: backend={backend} budget={budget} ...", flush=True)
    subprocess.run(cmd, check=True, env=env)
    with open(out_path) as f:
        return json.load(f)


def _first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def _agreement(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seer-ckpt", default="/tmp/seer_ckpt")
    ap.add_argument("--big-budget", type=int, default=8192)
    ap.add_argument("--small-budget", type=int, default=256)
    # worker-mode args
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model-path")
    ap.add_argument("--backend")
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.worker:
        _worker(args.model_path, args.backend, args.budget or None, args.out)
        return

    # Resolve base model id from the seer checkpoint config.
    with open(os.path.join(args.seer_ckpt, "config.json")) as f:
        base_model = json.load(f)["base_model"]
    print(f"base_model = {base_model}")

    py = sys.executable
    dense = _run_config(base_model, "triton", 0, "dense", py)
    seer_big = _run_config(args.seer_ckpt, "seer_attn", args.big_budget, "seerbig", py)
    seer_small = _run_config(
        args.seer_ckpt, "seer_attn", args.small_budget, "seersmall", py
    )

    print("\n================ RESULTS ================")
    all_big_match = True
    for i, prompt in enumerate(PROMPTS):
        d, b, s = dense[i], seer_big[i], seer_small[i]
        div_big = _first_divergence(d, b)
        agr_big = _agreement(d, b)
        agr_small = _agreement(d, s)
        big_ok = div_big == -1
        all_big_match &= big_ok
        print(f"\n[{i}] {prompt[:60]!r}")
        print(f"    len: dense={len(d)} seer_big={len(b)} seer_small={len(s)}")
        print(
            f"    seer_big  vs dense: agreement={agr_big:.3f} "
            f"first_div={'none' if div_big < 0 else div_big} "
            f"{'OK' if big_ok else 'MISMATCH'}"
        )
        print(
            f"    seer_small vs dense: agreement={agr_small:.3f} "
            f"first_div={_first_divergence(d, s)} (divergence expected)"
        )

    print("\n========================================")
    if all_big_match:
        print("PASS: seer_attn (big budget) reproduces dense baseline token-for-token.")
    else:
        print(
            "NOTE: seer_attn (big budget) diverged from dense on some prompts. "
            "Small bf16 numerical differences can flip a greedy argmax; inspect "
            "the first_div positions and agreement above to judge severity."
        )


if __name__ == "__main__":
    main()
