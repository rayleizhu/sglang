"""Reproduce SeerAttention-R evaluation (AIME24) with SGLang.

Runs AIME24 through SGLang under several attention configs and reports
accuracy using the *reference* answer-extraction + grading utilities
(``SeerAttention/eval/reasoning_tasks/Utils``), so the metric matches the
paper's pipeline.  Only the generation engine differs (SGLang instead of HF).

Configs compared:
  * dense        — base Qwen3 + triton backend (upper bound)
  * seer-<budget>— seer_attn backend at the given token budget(s)

Each config runs in its own subprocess (one Engine at a time).  Greedy decoding
(temperature 0) for reproducibility; the paper samples, but greedy gives a
stable single-run accuracy suitable for a dense-vs-sparse comparison.

Usage:
    python seer_attn_eval_aime.py \
        --seer-ckpt /tmp/seer_ckpt \
        --ref-repo /home/devuser/SeerAttention \
        --budgets 2048,4096 --max-new 8192 --limit -1
"""

import argparse
import json
import os
import subprocess
import sys


def _load_aime24(ref_repo):
    path = os.path.join(ref_repo, "eval/reasoning_tasks/data/aime24/test.jsonl")
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _build_prompts(examples, ckpt):
    from transformers import AutoTokenizer

    # The seer checkpoint ships only gate weights + config.json (no tokenizer);
    # fall back to the base model's tokenizer when ckpt has none.
    tok_src = ckpt
    if not os.path.exists(os.path.join(ckpt, "tokenizer_config.json")):
        with open(os.path.join(ckpt, "config.json")) as f:
            tok_src = json.load(f)["base_model"]
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    prompts = []
    for ex in examples:
        q = ex["problem"].strip()
        messages = [
            {
                "role": "user",
                "content": q
                + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
            }
        ]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    return prompts


# --------------------------------------------------------------------------
# Worker: run one config, write generated texts.
# --------------------------------------------------------------------------
def _worker(ckpt, ref_repo, backend, budget, max_new, limit, out_path):
    if backend == "reference":
        _worker_reference(ckpt, ref_repo, budget, max_new, limit, out_path)
        return

    # The seer checkpoint ships only gate weights + config.json (no tokenizer);
    # both prompt building and the Engine must read the tokenizer from the base
    # model in that case.
    with open(os.path.join(ckpt, "config.json")) as f:
        base_model = json.load(f)["base_model"]
    if backend == "seer_attn":
        os.environ["SGLANG_SEER_TOKEN_BUDGET"] = str(budget)
        os.environ["SGLANG_SEER_START_LAYER"] = "0"
        model_path = ckpt
        tokenizer_path = base_model
    else:
        # dense baseline uses the base model directly
        model_path = base_model
        tokenizer_path = base_model

    import sglang as sgl

    examples = _load_aime24(ref_repo)
    if limit > 0:
        examples = examples[:limit]
    prompts = _build_prompts(examples, ckpt)

    llm = sgl.Engine(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        attention_backend=backend,
        mem_fraction_static=0.6,
        max_running_requests=32,
        disable_cuda_graph=True,
        log_level="error",
    )
    outs = llm.generate(
        prompts,
        {"temperature": 0.0, "max_new_tokens": max_new},
    )
    llm.shutdown()

    records = []
    for ex, o in zip(examples, outs):
        records.append({"answer": str(ex["answer"]), "text": o["text"]})
    with open(out_path, "w") as f:
        json.dump(records, f)


# --------------------------------------------------------------------------
# Reference worker: official SeerAttention-R HF model (vendored seer_ref).
#
# Runs the SAME AIME24 prompts / greedy / max_new / grading as the SGLang
# worker, so the only difference is the generation engine — giving a true
# accuracy comparison between our seer_attn backend and the reference.
# --------------------------------------------------------------------------
def _worker_reference(ckpt, ref_repo, budget, max_new, limit, out_path):
    import torch

    # The vendored reference lives at <repo>/3rdparty/seerattention_ref; this
    # file is at <repo>/test/manual/attention/, so go up three levels.
    _REPO_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    sys.path.insert(0, os.path.join(_REPO_ROOT, "3rdparty"))
    from seerattention_ref.decode_sparse.qwen3.modeling_qwen3_seerattn_inference import (
        SeerDecodingQwen3ForCausalLM,
    )
    from transformers import AutoTokenizer

    with open(os.path.join(ckpt, "config.json")) as f:
        base_model = json.load(f)["base_model"]
    tok_src = ckpt
    if not os.path.exists(os.path.join(ckpt, "tokenizer_config.json")):
        tok_src = base_model
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    model = SeerDecodingQwen3ForCausalLM.from_pretrained(
        ckpt,
        load_gate=True,
        torch_dtype=torch.bfloat16,
        seerattn_implementation="seer_sparse",
        seerattn_sparsity_method="token_budget",
        seerattn_token_budget=budget,
        seerattn_start_layer=0,
    ).cuda().eval()

    examples = _load_aime24(ref_repo)
    if limit > 0:
        examples = examples[:limit]
    prompts = _build_prompts(examples, ckpt)

    records = []
    for ex, p in zip(examples, prompts):
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        attn = torch.ones_like(ids)
        with torch.no_grad():
            gen, _ = model.batch_exist_generate(
                ids,
                attention_mask=attn,
                max_length=ids.shape[1] + max_new,
                do_sample=False,
            )
        text = tok.decode(gen[0, ids.shape[1] :], skip_special_tokens=True)
        records.append({"answer": str(ex["answer"]), "text": text})
    with open(out_path, "w") as f:
        json.dump(records, f)


# --------------------------------------------------------------------------
# Scoring (reference Utils).
# --------------------------------------------------------------------------
def _score(records, ref_repo):
    sys.path.insert(0, os.path.join(ref_repo, "eval/reasoning_tasks"))
    from Utils.parser import extract_answer
    from Utils.grader import math_equal

    correct = 0
    for r in records:
        pred = extract_answer(r["text"], use_last_number=True)
        gt = r["answer"]
        try:
            ok = math_equal(pred, gt)
        except Exception:
            ok = False
        correct += int(ok)
    return correct, len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seer-ckpt", default="/tmp/seer_ckpt")
    ap.add_argument("--ref-repo", default="/home/devuser/SeerAttention")
    ap.add_argument("--budgets", default="2048,4096")
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument(
        "--with-reference",
        action="store_true",
        help="also run the official SeerAttention-R HF model (vendored seer_ref) "
        "at each budget, for a sglang-vs-reference accuracy comparison.",
    )
    # worker mode
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--backend")
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.worker:
        _worker(
            args.seer_ckpt,
            args.ref_repo,
            args.backend,
            args.budget,
            args.max_new,
            args.limit,
            args.out,
        )
        return

    py = sys.executable
    common = [
        "--seer-ckpt", args.seer_ckpt,
        "--ref-repo", args.ref_repo,
        "--max-new", str(args.max_new),
        "--limit", str(args.limit),
    ]

    configs = []
    if not args.skip_dense:
        configs.append(("dense", "triton", 0))
    for b in args.budgets.split(","):
        b = b.strip()
        if b:
            configs.append((f"seer-{b}", "seer_attn", int(b)))
            if args.with_reference:
                configs.append((f"ref-{b}", "reference", int(b)))

    summary = []
    for tag, backend, budget in configs:
        out_path = f"/tmp/aime_{tag}.json"
        print(f"[run] {tag} (backend={backend} budget={budget}) ...", flush=True)
        subprocess.run(
            [py, __file__, "--worker", "--backend", backend, "--budget",
             str(budget), "--out", out_path] + common,
            check=True,
        )
        records = json.load(open(out_path))
        correct, total = _score(records, args.ref_repo)
        acc = correct / total if total else 0.0
        summary.append((tag, correct, total, acc))
        print(f"     {tag}: {correct}/{total} = {acc:.3f}", flush=True)

    print("\n================ AIME24 accuracy ================")
    for tag, correct, total, acc in summary:
        print(f"  {tag:16s} {correct:3d}/{total:<3d}  acc={acc:.3f}")


if __name__ == "__main__":
    main()
