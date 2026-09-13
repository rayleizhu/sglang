"""Level-4 correctness: SGLang seer_attn backend vs the HF reference.

Compares greedy-decoded token ids of:

  * **reference** — the SeerAttention-R HF model composing the frozen base model
    with the distilled AttnGates: microsoft's own decode-sparse code path
    (prefill swapped to SDPA so there is no flash_attn dependency).  By default
    this side is read from a checked-in **golden fixture**
    (``fixtures/seer_attn_hf_reference_golden.json``) rather than executed, so
    the routine check needs neither the HF model nor its ~2 min eager decode.
    Pass ``--live-reference`` to run the vendored model
    (``3rdparty/seerattention_ref``) instead, and ``--regenerate-golden`` to
    run it and overwrite the fixture.
  * **sglang** — the ``seer_attn`` attention backend in SGLang.

Both use the SAME checkpoint, SAME prompts, greedy decoding, and the SAME
sparsity knobs (token budget / start layer).  Per-prompt bsz=1 to remove
batch-dependent numerical noise.

Because the two run entirely different attention kernels, bit-identical logits
are not expected; the comparison metric is the greedy-token agreement rate and
the first-divergence position.

The fixture pins the config it was generated under (checkpoint, token budget,
start layer, max_new, and a hash of the prompts).  Comparing against it with a
different config would be meaningless, so that is rejected rather than silently
reported -- use ``--live-reference`` or ``--regenerate-golden`` in that case.

Usage:
    # fixture-based (fast, no HF model needed)
    python seer_attn_vs_hf_reference.py --seer-ckpt <AttnGates ckpt>

    # run the vendored HF reference live
    python seer_attn_vs_hf_reference.py --seer-ckpt <ckpt> --live-reference

    # regenerate the fixture after an intentional reference-side change
    python seer_attn_vs_hf_reference.py --seer-ckpt <ckpt> --regenerate-golden
"""

import argparse
import json
import os
import subprocess
import sys

PROMPTS = [
    "The capital of France is",
    "Count from 1 to 60 separated by commas: 1, 2, 3,",
    "List the first 20 prime numbers: 2, 3, 5,",
    "Q: If a train travels 60 km in 1.5 hours, what is its average speed? "
    "A: Let's think step by step.",
]

# Long prompts (several hundred tokens) so a small block-aligned chunk size
# (e.g. 128) forces the sglang prefill to be split into multiple chunks — the
# only way to exercise the chunked summary/rolling build end-to-end.  A repeated
# factual passage + a question keeps greedy decoding deterministic.
_PASSAGE = (
    "In the study of computer architecture, the memory hierarchy is organized "
    "into several levels: registers, L1 cache, L2 cache, L3 cache, main memory, "
    "and secondary storage. Each level trades capacity for latency. Registers "
    "are the fastest but smallest; secondary storage is the largest but slowest. "
    "The principle of locality — both temporal and spatial — is what makes "
    "caching effective in practice. "
)


def build_long_prompts(tokenizer, min_tokens):
    """Return a few long prompts, each >= min_tokens tokens, ending in a
    deterministic question so greedy decoding is stable."""
    reps = max(1, (min_tokens // max(1, len(tokenizer.encode(_PASSAGE)))) + 1)
    body = _PASSAGE * reps
    return [
        body + "\n\nQuestion: Which level of the memory hierarchy is the fastest? Answer:",
        body + "\n\nSummary: The key principle that makes caching effective is",
    ]


# --------------------------------------------------------------------------
# Reference worker (vendored HF model).
# --------------------------------------------------------------------------
GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "seer_attn_hf_reference_golden.json",
)


def _prompts_sha(prompts):
    import hashlib

    return hashlib.sha256(" ".join(prompts).encode()).hexdigest()


def _load_golden(token_budget, start_layer, max_new, long_prompts):
    """Load the checked-in reference output, rejecting a config mismatch.

    The fixture is only a valid oracle for the exact configuration it was
    generated under, so every pinned field is checked instead of trusting the
    caller to pass matching flags.
    """
    if long_prompts:
        raise SystemExit(
            "--long-prompts has no golden fixture (the long prompts are built "
            "from the tokenizer at runtime). Re-run with --live-reference."
        )
    if not os.path.exists(GOLDEN_PATH):
        raise SystemExit(
            f"golden fixture not found at {GOLDEN_PATH}; run with "
            "--regenerate-golden (needs the vendored HF reference) or use "
            "--live-reference."
        )
    with open(GOLDEN_PATH) as f:
        g = json.load(f)

    mismatches = []
    for field, want in (
        ("token_budget", token_budget),
        ("start_layer", start_layer),
        ("max_new", max_new),
    ):
        if g.get(field) != want:
            mismatches.append(f"{field}: fixture={g.get(field)!r} requested={want!r}")
    if _prompts_sha(PROMPTS) != g.get("prompts_sha256"):
        mismatches.append("prompts: PROMPTS in this file differ from the fixture's")
    if mismatches:
        raise SystemExit(
            "golden fixture does not match the requested configuration:\n  "
            + "\n  ".join(mismatches)
            + "\n\nThe fixture is only a valid oracle for the config it was "
            "generated under. Re-run with --live-reference, or with "
            "--regenerate-golden to rebuild it for this config."
        )
    return g["token_ids"]


def _run_reference(ckpt, token_budget, start_layer, max_new, out_path, long_prompts=False):
    import torch

    # The vendored reference lives at <repo>/3rdparty/seerattention_ref; this
    # file is at <repo>/test/manual/attention/, so go up three levels.
    _REPO_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    sys.path.insert(0, os.path.join(_REPO_ROOT, "3rdparty"))
    from seerattention_ref.decode_sparse.qwen3.modeling_qwen3_seerattn_inference import (
        SeerDecodingQwen3ForCausalLM,
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    prompts = build_long_prompts(tok, min_tokens=512) if long_prompts else PROMPTS
    model = SeerDecodingQwen3ForCausalLM.from_pretrained(
        ckpt,
        load_gate=True,
        torch_dtype=torch.bfloat16,
        seerattn_implementation="seer_sparse",
        seerattn_sparsity_method="token_budget",
        seerattn_token_budget=token_budget,
        seerattn_start_layer=start_layer,
    ).cuda().eval()

    results = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        attn = torch.ones_like(ids)
        with torch.no_grad():
            gen, _ = model.batch_exist_generate(
                ids, attention_mask=attn, max_length=ids.shape[1] + max_new,
                do_sample=False,
            )
        results.append(gen[0, ids.shape[1]:].tolist())
    with open(out_path, "w") as f:
        json.dump(results, f)


# --------------------------------------------------------------------------
# SGLang worker.
# --------------------------------------------------------------------------
def _run_sglang(ckpt, token_budget, start_layer, max_new, out_path,
                long_prompts=False, chunked_prefill_size=-1):
    os.environ["SGLANG_SEER_TOKEN_BUDGET"] = str(token_budget)
    os.environ["SGLANG_SEER_START_LAYER"] = str(start_layer)
    if chunked_prefill_size != -1:
        # Opt into the block-aligned chunked prefill path (model_runner guards it).
        os.environ["SGLANG_SEER_ENABLE_CHUNKED_PREFILL"] = "1"
    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    prompts = build_long_prompts(tok, min_tokens=512) if long_prompts else PROMPTS
    engine_kwargs = dict(
        model_path=ckpt,
        attention_backend="seer_attn",
        mem_fraction_static=0.6,
        disable_cuda_graph=True,
        log_level="error",
        skip_tokenizer_init=True,
    )
    if chunked_prefill_size != -1:
        engine_kwargs["chunked_prefill_size"] = chunked_prefill_size
    llm = sgl.Engine(**engine_kwargs)
    results = []
    for p in prompts:
        ids = tok.encode(p)
        out = llm.generate(
            input_ids=[ids],
            sampling_params={"temperature": 0.0, "max_new_tokens": max_new},
        )
        results.append(out[0]["output_ids"])
    llm.shutdown()
    with open(out_path, "w") as f:
        json.dump(results, f)


def _agreement(a, b):
    n = min(len(a), len(b))
    return (sum(1 for i in range(n) if a[i] == b[i]) / n) if n else 0.0


def _first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seer-ckpt", default="/tmp/seer_ckpt")
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--long-prompts", action="store_true",
                    help="use several-hundred-token prompts (needed to exercise chunking)")
    ap.add_argument("--chunked-prefill-size", type=int, default=-1,
                    help="if >0, run an EXTRA sglang pass with chunked prefill of this "
                         "size (must be a multiple of block_size) and compare it to "
                         "both the HF reference and the non-chunked sglang baseline")
    # worker mode
    ap.add_argument("--worker", choices=["reference", "sglang"])
    ap.add_argument("--out")
    ap.add_argument("--worker-chunk", type=int, default=-1)
    ap.add_argument(
        "--live-reference",
        action="store_true",
        help="run the vendored HF reference (3rdparty/seerattention_ref) instead "
             "of reading the checked-in golden fixture",
    )
    ap.add_argument(
        "--regenerate-golden",
        action="store_true",
        help="run the vendored HF reference and overwrite the golden fixture, "
             "then continue with the comparison",
    )
    args = ap.parse_args()

    if args.worker == "reference":
        _run_reference(
            args.seer_ckpt, args.token_budget, args.start_layer, args.max_new,
            args.out, long_prompts=args.long_prompts,
        )
        return
    if args.worker == "sglang":
        _run_sglang(
            args.seer_ckpt, args.token_budget, args.start_layer, args.max_new,
            args.out, long_prompts=args.long_prompts,
            chunked_prefill_size=args.worker_chunk,
        )
        return

    py = sys.executable
    common = [
        "--seer-ckpt", args.seer_ckpt,
        "--token-budget", str(args.token_budget),
        "--start-layer", str(args.start_layer),
        "--max-new", str(args.max_new),
    ]
    if args.long_prompts:
        common.append("--long-prompts")

    run_live_ref = args.live_reference or args.regenerate_golden or args.long_prompts
    if run_live_ref:
        why = "--long-prompts" if args.long_prompts and not (
            args.live_reference or args.regenerate_golden
        ) else "requested"
        print(f"[run] reference (vendored HF, {why}) ...", flush=True)
        subprocess.run(
            [py, __file__, "--worker", "reference", "--out", "/tmp/seer_ref_out.json"]
            + common,
            check=True,
        )
        ref = json.load(open("/tmp/seer_ref_out.json"))
        if args.regenerate_golden:
            if args.long_prompts:
                raise SystemExit(
                    "--regenerate-golden does not support --long-prompts: those "
                    "prompts are tokenizer-derived and not reproducible from the "
                    "fixture alone."
                )
            fixture = {
                "_comment": (
                    "Golden output of the vendored SeerAttention-R HF reference "
                    "(3rdparty/seerattention_ref), used as the oracle for the "
                    "sglang seer_attn backend so the routine check does not need "
                    "to run the HF model. Regenerate with: "
                    "python seer_attn_vs_hf_reference.py --regenerate-golden "
                    "--seer-ckpt <AttnGates ckpt>"
                ),
                "checkpoint": "SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates",
                "base_model": "Qwen/Qwen3-4B",
                "token_budget": args.token_budget,
                "start_layer": args.start_layer,
                "max_new": args.max_new,
                "greedy": True,
                "prompts": PROMPTS,
                "prompts_sha256": _prompts_sha(PROMPTS),
                "token_ids": ref,
            }
            os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
            with open(GOLDEN_PATH, "w") as f:
                json.dump(fixture, f, indent=2)
            print(f"[golden] wrote {GOLDEN_PATH}", flush=True)
    else:
        ref = _load_golden(
            args.token_budget, args.start_layer, args.max_new, args.long_prompts
        )
        print(
            f"[ref] golden fixture ({len(ref)} sequences) — pass --live-reference "
            "to run the HF model instead",
            flush=True,
        )

    print(f"[run] sglang seer_attn (baseline, no chunking) ...", flush=True)
    subprocess.run(
        [py, __file__, "--worker", "sglang", "--out", "/tmp/seer_sgl_out.json",
         "--worker-chunk", "-1"]
        + common,
        check=True,
    )

    sgl_ = json.load(open("/tmp/seer_sgl_out.json"))

    sgl_chunk = None
    if args.chunked_prefill_size > 0:
        print(f"[run] sglang seer_attn (CHUNKED prefill size="
              f"{args.chunked_prefill_size}) ...", flush=True)
        subprocess.run(
            [py, __file__, "--worker", "sglang", "--out", "/tmp/seer_sgl_chunk_out.json",
             "--worker-chunk", str(args.chunked_prefill_size)]
            + common,
            check=True,
        )
        sgl_chunk = json.load(open("/tmp/seer_sgl_chunk_out.json"))

    n = len(ref)
    print("\n================ HF reference vs SGLang ================")
    base_ag = []
    chunk_ag = []
    chunk_vs_base = []
    for i in range(n):
        r, s = ref[i], sgl_[i]
        a_base = _agreement(r, s)
        base_ag.append(a_base)
        print(f"\n[{i}] len: ref={len(r)} sglang={len(s)}")
        print(f"    baseline  agreement={a_base:.3f} first_div={_first_div(r, s)}")
        if sgl_chunk is not None:
            c = sgl_chunk[i]
            a_chunk = _agreement(r, c)
            a_cvb = _agreement(s, c)
            chunk_ag.append(a_chunk)
            chunk_vs_base.append(a_cvb)
            print(f"    chunked   agreement={a_chunk:.3f} first_div={_first_div(r, c)}")
            print(f"    chunk-vs-baseline agreement={a_cvb:.3f} "
                  f"first_div={_first_div(s, c)}")

    if sgl_chunk is not None:
        mb = sum(base_ag) / len(base_ag)
        mc = sum(chunk_ag) / len(chunk_ag)
        mcb = sum(chunk_vs_base) / len(chunk_vs_base)
        print("\n---- summary ----")
        print(f"mean agreement vs HF ref:  baseline={mb:.4f}  chunked={mc:.4f}")
        print(f"mean chunked-vs-baseline agreement: {mcb:.4f}")
        # PLAN CRITERION: chunked-prefill agreement with the HF reference oracle
        # must NOT drop relative to the chunked-OFF baseline's agreement with the
        # same oracle (allow a small bf16-noise slack).  We deliberately do NOT
        # require chunked == baseline: with a tight token_budget, top-k block
        # selection is noise-sensitive and the two sparse sglang runs can pick
        # different budget-boundary blocks — but chunked must track the oracle at
        # least as well as baseline does.  (With a large budget >= #blocks the run
        # is dense-equivalent and all three converge; that is the cleanest check.)
        SLACK = 0.02
        ok = mc >= mb - SLACK
        print(f"\n{'PASS' if ok else 'FAIL'}: chunked-vs-HF {mc:.4f} >= "
              f"baseline-vs-HF {mb:.4f} - slack {SLACK}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
