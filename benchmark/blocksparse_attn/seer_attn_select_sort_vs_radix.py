"""Micro-benchmark: sort-based vs radix-select threshold in block selection.

B-plan A/B.  Times the production sort kernel (_select_from_scores_kernel)
against the sort-free radix-select kernel (_select_from_scores_radix_kernel),
which are byte-identical in output (see seer_attn_radix_parity.py) and differ
only in how the n_pick-th-largest threshold is found:

  sort : tl.sort(acc)            -> O(NBLK·log²NBLK) bitonic
  radix: 32-pass bit-serial select -> O(32·NBLK), no log² factor

The question is the NBLK-scaling crossover: at small NBLK the kernel is
launch-bound (both ~equal); the sort path's log² cost only dominates at large
NBLK (long context).  block_size=64 so NBLK=N ~ N*64 token context.

Usage:
    .pixi/envs/default/bin/python test/srt/seer_attn_select_sort_vs_radix.py
"""

import torch

from sglang.srt.layers.attention.blocksparse.common_index_kernels import (
    _select_from_scores_kernel,
    _select_from_scores_radix_kernel,
)


def _bench(kernel, bsz, H, NBLK, topk, block_size, num_init, num_local, iters=1000):
    device = "cuda"
    n_pick = topk - (num_init + num_local)
    torch.manual_seed(0)
    scores = torch.randn(bsz, H, NBLK, dtype=torch.float32, device=device)
    # num_prev = NBLK-1: every block id a candidate (full selection work).
    seq_lens = torch.full((bsz,), (NBLK - 1) * block_size + 1, dtype=torch.int32, device=device)
    active = torch.empty(bsz, H, topk, dtype=torch.int32, device=device)

    def run():
        kernel[(bsz, H)](
            scores, scores.stride(0), scores.stride(1),
            seq_lens,
            active, active.stride(0), active.stride(1),
            block_size=block_size, topk=topk, n_pick=n_pick,
            num_init_blocks=num_init, num_local_blocks=num_local, NBLK=NBLK,
        )

    for _ in range(50):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us/call


def main():
    H = 8
    block_size = 64
    topk = 32
    num_init, num_local = 1, 0
    shapes = [128, 512, 1024, 2048, 4096, 8192, 16384]
    batches = [1, 8, 32]

    print(f"H={H} block_size={block_size} topk={topk} init={num_init} local={num_local}")
    print(f"{'NBLK':>6} {'bsz':>4} {'sort us':>9} {'radix us':>9} {'speedup':>8}")
    print("-" * 42)
    for NBLK in shapes:
        for bsz in batches:
            try:
                t_sort = _bench(_select_from_scores_kernel, bsz, H, NBLK, topk, block_size, num_init, num_local)
                t_radix = _bench(_select_from_scores_radix_kernel, bsz, H, NBLK, topk, block_size, num_init, num_local)
                sp = t_sort / t_radix if t_radix > 0 else float("nan")
                print(f"{NBLK:6d} {bsz:4d} {t_sort:9.3f} {t_radix:9.3f} {sp:7.3f}x")
            except Exception as e:
                msg = str(e).strip().splitlines()[-1][:50] if str(e).strip() else type(e).__name__
                print(f"{NBLK:6d} {bsz:4d}   FAILED: {msg}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
