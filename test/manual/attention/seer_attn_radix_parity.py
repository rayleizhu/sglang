"""Correctness parity: radix-select threshold kernel vs the sort-based kernel.

B-plan (_select_from_scores_radix_kernel) must produce BYTE-IDENTICAL output to
the production _select_from_scores_kernel on every (req, head) row, across NBLK,
n_pick, and -inf-padding regimes.  Both share the same downstream gt/eq quota
fill + ascending compaction; only the threshold-finding differs (radix vs sort).

We drive each kernel directly on a synthetic [bsz, H, NBLK] scores tensor (the
same one the score kernel would emit) and compare the [bsz, H, topk] active ids.

Usage:
    .pixi/envs/default/bin/python test/srt/seer_attn_radix_parity.py
"""

import torch
import triton

from sglang.srt.layers.attention.blocksparse.common_index_kernels import (
    _select_from_scores_kernel,
    _select_from_scores_radix_kernel,
)


def _run(kernel, scores, seq_lens, *, block_size, topk, n_pick, num_init, num_local, NBLK):
    bsz, H, _ = scores.shape
    active = torch.empty(bsz, H, topk, dtype=torch.int32, device=scores.device)
    kernel[(bsz, H)](
        scores, scores.stride(0), scores.stride(1),
        seq_lens,
        active, active.stride(0), active.stride(1),
        block_size=block_size, topk=topk, n_pick=n_pick,
        num_init_blocks=num_init, num_local_blocks=num_local, NBLK=NBLK,
    )
    return active


def _make_scores(bsz, H, NBLK, num_prev, dup=False, device="cuda", seed=0):
    """Scores [bsz,H,NBLK]: candidate lanes [0,num_prev) finite, rest -inf.

    dup=True forces many exact ties (quantized-summary regime) to stress the
    gt/eq threshold-tie path.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    scores = torch.full((bsz, H, NBLK), float("-inf"), dtype=torch.float32, device=device)
    vals = torch.randn(bsz, H, num_prev, generator=g, device=device)
    if dup:
        vals = (vals * 3).round() / 3  # collapse to a few distinct levels -> ties
    scores[:, :, :num_prev] = vals
    return scores


def main():
    device = "cuda"
    H = 8
    block_size = 64
    cases = []
    # (NBLK, topk, num_init, num_local, num_prev, dup)
    for NBLK in [128, 512, 1024, 4096, 16384]:
        for topk in [32, 16]:
            for num_prev in [NBLK - 1, topk + 5, topk - 3, NBLK // 2, 1]:
                if num_prev <= 0:
                    continue
                cases.append((NBLK, topk, 1, 0, num_prev, False))
                cases.append((NBLK, topk, 1, 0, num_prev, True))  # tie stress
        # forced-band variants
        cases.append((NBLK, 32, 4, 4, NBLK - 1, False))
        cases.append((NBLK, 32, 0, 0, NBLK - 1, False))

    n_fail = 0
    n_ok = 0
    for NBLK, topk, num_init, num_local, num_prev, dup in cases:
        n_pick = topk - (num_init + num_local)
        bsz = 4
        scores = _make_scores(bsz, H, NBLK, num_prev, dup=dup, seed=NBLK + topk + num_prev)
        seq_lens = torch.full((bsz,), num_prev * block_size + 1, dtype=torch.int32, device=device)

        a_sort = _run(_select_from_scores_kernel, scores, seq_lens,
                      block_size=block_size, topk=topk, n_pick=n_pick,
                      num_init=num_init, num_local=num_local, NBLK=NBLK)
        a_radix = _run(_select_from_scores_radix_kernel, scores, seq_lens,
                       block_size=block_size, topk=topk, n_pick=n_pick,
                       num_init=num_init, num_local=num_local, NBLK=NBLK)

        if torch.equal(a_sort, a_radix):
            n_ok += 1
        else:
            n_fail += 1
            mism = (a_sort != a_radix).sum().item()
            print(f"MISMATCH NBLK={NBLK} topk={topk} init={num_init} local={num_local} "
                  f"num_prev={num_prev} dup={dup}: {mism} elems differ")
            # show first mismatching row
            bad = (a_sort != a_radix).any(dim=-1).nonzero()[0]
            bi, hi = bad[0].item(), bad[1].item()
            print(f"  sort [{bi},{hi}]: {a_sort[bi,hi].tolist()}")
            print(f"  radix[{bi},{hi}]: {a_radix[bi,hi].tolist()}")

    print(f"\n{n_ok} cases OK, {n_fail} failed")


if __name__ == "__main__":
    main()
