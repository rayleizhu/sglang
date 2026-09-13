"""Unit tests for the SeerAttention-R decode-sparse attention backend.

These are self-contained numerical tests (no network / no checkpoint):

* ``test_rolling_accumulator_fold`` — the rolling accumulator folds each new
  key into a per-(req, kv_head) running [max | min | sum], self-resetting on the
  first token of a block.
* ``test_rolling_accumulator_ignores_padding_rows`` — CUDA-graph padding rows
  (seq_len==1, req_pool_idx==0) must be no-ops, not corrupt pool slot 0.
* ``test_block_selection_budget`` — token-budget block selection keeps at most
  ``block_budget`` top-scoring blocks plus the forced last block.

The block-sparse decode itself is the shared flashinfer virtual-batch path
(covered by integration tests), not a standalone kernel.

Usage:
    python3 -m unittest test_seer_attn_backend
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Self-contained kernel/gate numerical tests for the SeerAttention-R backend.
register_cuda_ci(est_time=60, suite="stage-b-test-small-1-gpu")

_HAS_CUDA = torch.cuda.is_available()


@unittest.skipUnless(_HAS_CUDA, "SeerAttention kernels require CUDA")
class TestSeerAttnKernels(CustomTestCase):
    def _dummy_decode_args(self, Hk, D, GATE_D, max_reqs, dev, dt):
        """Minimal gate/summary tensors for update_summary_cache_decode when only
        the rolling-fold path is exercised (schedules avoiding block fills)."""
        max_blocks = 8
        req_to_summary = torch.arange(
            max_reqs * max_blocks, device=dev, dtype=torch.int64
        ).view(max_reqs, max_blocks)
        summary_cache = torch.zeros(
            max_reqs * max_blocks + 1, Hk, GATE_D, device=dev, dtype=dt
        )
        proj_weight = torch.zeros(Hk, 3 * D, GATE_D, device=dev, dtype=dt)
        norm_weight = torch.ones(GATE_D, device=dev, dtype=dt)
        inv_freq = 1.0 / (
            1_000_000.0
            ** (torch.arange(0, GATE_D, 2, dtype=torch.float32, device=dev) / GATE_D)
        )
        return req_to_summary, summary_cache, proj_weight, norm_weight, inv_freq

    def test_rolling_accumulator_fold(self):
        # The rolling accumulator folds each new key into a per-(req, kv_head)
        # running [max | min | sum] (fp32, width 3*D), self-resetting on the
        # first token of a block (pos_in_block == 0).  Exercised via the fused
        # update_summary_cache_decode on schedules that avoid block fills (so the
        # summary-compression branch never fires — only the fold is tested).
        from sglang.srt.layers.attention.blocksparse.seer_attn.cache_kernels import (
            update_summary_cache_decode,
        )

        torch.manual_seed(0)
        dev = "cuda"
        dt = torch.bfloat16
        bsz, Hk, D, block_size = 3, 2, 128, 64
        GATE_D = D
        max_reqs = 8
        rolling = torch.zeros(max_reqs, Hk, 3 * D, device=dev, dtype=torch.float32)
        req_pool_indices = torch.tensor([0, 1, 2], device=dev)
        rts, summ, pw, nw, ifr = self._dummy_decode_args(
            Hk, D, GATE_D, max_reqs, dev, dt
        )

        # Reference accumulators (one running reduction per req over its block).
        ref = {b: None for b in range(bsz)}
        # Each request's FIRST step lands on a block boundary
        # ((seq_len - 1) % block_size == 0) so the kernel self-resets the
        # accumulator from the new key (the realistic flow — in production the
        # accumulator is seeded by prefill or reset on a boundary, never folded
        # into raw zeros mid-block).  Subsequent steps fold.  All seq_lens are
        # >= 2 (a real decode step always has >=1 prefilled token + the current
        # one); seq_len == 1 is reserved for graph-padding rows, which the fold
        # kernel skips (see test_rolling_accumulator_ignores_padding_rows).
        # None is ≡ 0 (mod 64), so no block fill -> summary branch never fires.
        seq_schedules = [
            torch.tensor([65, 129, 193], dtype=torch.int32, device=dev),
            torch.tensor([66, 130, 194], dtype=torch.int32, device=dev),
            torch.tensor([67, 131, 195], dtype=torch.int32, device=dev),
        ]
        for seqlens in seq_schedules:
            knew = torch.randn(bsz, Hk, D, device=dev, dtype=dt)
            update_summary_cache_decode(
                knew, rolling, seqlens, req_pool_indices, rts, summ, pw, nw, ifr,
                block_size=block_size,
            )
            for b in range(bsz):
                kf = knew[b].float()  # [Hk, D]
                first = (int(seqlens[b]) - 1) % block_size == 0
                if first or ref[b] is None:
                    ref[b] = [kf.clone(), kf.clone(), kf.clone()]  # max,min,sum
                else:
                    ref[b][0] = torch.maximum(ref[b][0], kf)
                    ref[b][1] = torch.minimum(ref[b][1], kf)
                    ref[b][2] = ref[b][2] + kf
            for b in range(bsz):
                got = rolling[req_pool_indices[b]]  # [Hk, 3*D]
                exp = torch.cat(ref[b], dim=-1)  # [Hk, 3*D]
                diff = (got - exp).abs().max().item()
                self.assertLess(diff, 1e-3, f"req {b} accumulator diff {diff}")

    def test_rolling_accumulator_ignores_padding_rows(self):
        # Regression: under CUDA-graph replay a real batch is padded up to the
        # captured bs.  Padding rows carry seq_len == fill_value (1) and
        # req_pool_indices == 0 (the zero-init graph buffer is only overwritten
        # for the raw_bs real rows).  If the fold wrote unconditionally, a
        # padding row would fold its garbage key into acc[req_pool_idx=0] and
        # race the *real* request that owns pool slot 0 — silently corrupting it.
        # The seq_len > 1 guard must make padding rows no-ops.
        from sglang.srt.layers.attention.blocksparse.seer_attn.cache_kernels import (
            update_summary_cache_decode,
        )

        torch.manual_seed(0)
        dev = "cuda"
        dt = torch.bfloat16
        Hk, D, block_size = 2, 128, 64
        GATE_D = D
        max_reqs = 8
        rolling = torch.zeros(max_reqs, Hk, 3 * D, device=dev, dtype=torch.float32)
        rts, summ, pw, nw, ifr = self._dummy_decode_args(
            Hk, D, GATE_D, max_reqs, dev, dt
        )

        # raw_bs = 1 real request occupying pool slot 0, padded to bs = 4.  The
        # 3 padding rows all carry seq_len = 1 and req_pool_idx = 0 (collide
        # with the real request's slot).  seq_len=130 -> not a block fill.
        bs = 4
        seqlens = torch.tensor([130, 1, 1, 1], dtype=torch.int32, device=dev)
        req_pool_indices = torch.tensor([0, 0, 0, 0], device=dev)
        knew = torch.randn(bs, Hk, D, device=dev, dtype=dt)

        update_summary_cache_decode(
            knew, rolling, seqlens, req_pool_indices, rts, summ, pw, nw, ifr,
            block_size=block_size,
        )

        # The real request (seq_len=130, pos_in_block=1 -> fold, not first) must
        # see exactly its own key folded onto the zero-init buffer; the padding
        # rows must not have touched acc[0].  Since slot 0 started at zeros and
        # only row 0 (a non-first fold) ran, acc = [max(0,k) | min(0,k) | 0+k].
        k0 = knew[0].float()  # [Hk, D]
        exp = torch.cat(
            [torch.maximum(torch.zeros_like(k0), k0),
             torch.minimum(torch.zeros_like(k0), k0),
             k0],
            dim=-1,
        )
        diff = (rolling[0] - exp).abs().max().item()
        self.assertLess(diff, 1e-3, f"padding rows corrupted slot 0: diff {diff}")
        # All other pool slots stay zero (padding wrote nowhere else either).
        self.assertEqual(rolling[1:].abs().max().item(), 0.0)

    def test_block_selection_budget(self):
        from sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate import (
            select_blocks_from_scores,
        )

        torch.manual_seed(0)
        dev = "cuda"
        bsz, Hk, nblk = 2, 2, 5
        budget = 3
        attn = torch.rand(bsz, Hk, nblk, device=dev)
        valid = torch.ones(bsz, Hk, nblk, dtype=torch.bool, device=dev)

        mask = select_blocks_from_scores(
            attn, valid, "token_budget", 0.0, budget
        )
        counts = mask.sum(-1)
        # topk(budget) + forced last block => at most budget + 1.
        self.assertTrue(bool((counts <= budget + 1).all()))
        # last block always selected.
        self.assertTrue(bool(mask[:, :, -1].all()))

    def test_block_selection_forces_last_valid_per_request(self):
        # Regression: in a batch the block axis is padded to the longest
        # request's block count.  The forced "most recent block" must be each
        # request's last *valid* block, NOT the global last column — otherwise a
        # shorter request selects an out-of-range block (whose tokens are all
        # beyond cache_seqlens), the sparse-decode kernel sees an all-empty row,
        # and the output degenerates to NaN.
        from sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate import (
            select_blocks_from_scores,
        )

        torch.manual_seed(0)
        dev = "cuda"
        bsz, Hk, nblk = 2, 2, 6
        budget = 2
        attn = torch.rand(bsz, Hk, nblk, device=dev)
        valid = torch.ones(bsz, Hk, nblk, dtype=torch.bool, device=dev)
        # Request 0 is short: only blocks 0..2 are valid (last valid = block 2).
        valid[0, :, 3:] = False

        mask = select_blocks_from_scores(attn, valid, "token_budget", 0.0, budget)

        # No invalid block may ever be selected.
        self.assertFalse(bool((mask & ~valid).any()))
        # Every (req, head) row selects at least one block (its last valid one).
        self.assertTrue(bool((mask.sum(-1) >= 1).all()))
        # Request 0's forced block is block 2 (its last valid), not column -1.
        self.assertTrue(bool(mask[0, :, 2].all()))
        self.assertFalse(bool(mask[0, :, 5].any()))
        # Request 1 (full length) forces the global last block.
        self.assertTrue(bool(mask[1, :, 5].all()))


@unittest.skipUnless(_HAS_CUDA, "AttnGate parity requires CUDA")
class TestSeerAttnGate(CustomTestCase):
    def test_gate_shapes(self):
        from sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate import AttnGate

        g = AttnGate(
            block_size=64,
            model_hidden_size=128,
            gate_hidden_size=128,
            num_k_head=8,
            num_q_head=32,
        ).cuda()
        kb = torch.randn(3, 64, 8, 128, device="cuda")
        pooled = g.pool_block_keys(kb)
        self.assertEqual(tuple(pooled.shape), (3, 8, 128 * 3))
        comp = g.compress_k(pooled)
        self.assertEqual(tuple(comp.shape), (3, 8, 128))
        q = torch.randn(2, 32, 128, device="cuda")
        qg = g.project_q(q)
        self.assertEqual(tuple(qg.shape), (2, 8, 128))


@unittest.skipUnless(_HAS_CUDA, "Prefill cache kernels require CUDA")
class TestSeerAttnPrefillCacheKernels(CustomTestCase):
    """The prefill summary/rolling Triton kernels must match the PyTorch gate
    path used by ``SeerAttnBackend._build_summaries_extend`` (pool -> linear_k
    -> knorm -> NeoX RoPE at block-start positions; raw tail copy for rolling).
    Uses a ragged batch (mixed complete + partial trailing blocks)."""

    def _rope_tables(self, rope_dim, theta, max_pos, device, dtype):
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
                / rope_dim
            )
        )
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype), inv_freq

    def test_prefill_summary_and_rolling_match_gate(self):
        from sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate import (
            AttnGate,
            apply_rotary_pos_emb_seer,
        )
        from sglang.srt.layers.attention.blocksparse.seer_attn.cache_kernels import (
            build_prefill_summary_schedule,
            update_summary_cache_prefill,
        )

        torch.manual_seed(0)
        dev, dt = "cuda", torch.bfloat16
        block_size, Hk, D, Hq = 64, 4, 128, 16
        GATE_D = D
        theta, max_pos = 1_000_000.0, 8192

        # Ragged: complete-only, complete+partial, exact one block, partial-only.
        ext = torch.tensor([200, 130, 64, 17], dtype=torch.int64, device=dev)
        bsz = ext.shape[0]
        num_tokens = int(ext.sum())
        k_nope = torch.randn(num_tokens, Hk, D, device=dev, dtype=dt)
        req_pool_indices = torch.tensor([3, 1, 0, 2], device=dev)

        max_blocks = int((ext // block_size).max()) + 2
        max_reqs = 8
        req_to_summary = torch.arange(
            max_reqs * max_blocks, device=dev, dtype=torch.int64
        ).view(max_reqs, max_blocks)
        summary_pool = max_reqs * max_blocks + 1

        gate = AttnGate(
            block_size=block_size,
            model_hidden_size=D,
            gate_hidden_size=GATE_D,
            num_k_head=Hk,
            num_q_head=Hq,
        ).to(dev)
        with torch.no_grad():
            gate.attngate_linear_k.weight.normal_(0, 0.05)
            gate.attngate_knorm.weight.uniform_(0.5, 1.5)
        gate = gate.to(dt)

        cos_tab, sin_tab, inv_freq = self._rope_tables(
            GATE_D, theta, max_pos, dev, dt
        )

        summary_k = torch.zeros(summary_pool, Hk, GATE_D, device=dev, dtype=dt)
        # Rolling accumulator: [max_reqs, Hk, 3*D] fp32 ([max | min | sum]).
        rolling_k = torch.zeros(max_reqs, Hk, 3 * D, device=dev, dtype=torch.float32)
        schedule = build_prefill_summary_schedule(
            ext, req_pool_indices, block_size, extend_seq_lens_cpu=ext.cpu()
        )
        update_summary_cache_prefill(
            schedule,
            k_nope,
            summary_k,
            rolling_k,
            req_to_summary,
            gate.attngate_linear_k.weight,
            gate.attngate_knorm.weight,
            inv_freq,
            block_size=block_size,
        )

        # Reference (PyTorch gate path).
        summary_ref = torch.zeros_like(summary_k)
        rolling_ref = torch.zeros_like(rolling_k)
        start = 0
        rpi = req_pool_indices.tolist()
        for i in range(bsz):
            L = int(ext[i])
            k_req = k_nope[start : start + L]
            start += L
            nf, rem = L // block_size, L % block_size
            if nf > 0:
                kb = k_req[: nf * block_size].view(nf, block_size, Hk, D)
                comp = gate.compress_k(gate.pool_block_keys(kb))
                pos = torch.arange(nf, device=dev) * block_size
                comp = apply_rotary_pos_emb_seer(
                    comp, cos_tab[pos].unsqueeze(1), sin_tab[pos].unsqueeze(1)
                )
                slots = req_to_summary[rpi[i], torch.arange(nf, device=dev)]
                summary_ref[slots] = comp.to(dt)
            if rem > 0:
                tail = k_req[nf * block_size :].float()  # [rem, Hk, D]
                kmax = tail.amax(0)
                kmin = tail.amin(0)
                ksum = tail.sum(0)
                rolling_ref[rpi[i]] = torch.cat([kmax, kmin, ksum], dim=-1)

        # Rolling accumulator: reductions in fp32 -> match closely (max/min are
        # order-independent; sum has only tiny fp accumulation-order diff).
        self.assertLess(
            (rolling_k - rolling_ref).abs().max().item(), 1e-3
        )
        # Summary: kernel accumulates in fp32, reference einsum in bf16 -> tiny
        # rounding diff only.  Compare only the written slots.
        s_diff = 0.0
        for i in range(bsz):
            for j in range(int(ext[i]) // block_size):
                slot = int(req_to_summary[rpi[i], j])
                s_diff = max(
                    s_diff,
                    (summary_k[slot].float() - summary_ref[slot].float())
                    .abs()
                    .max()
                    .item(),
                )
        self.assertLess(s_diff, 3e-2, f"summary diff {s_diff}")

    def test_decode_summary_update_matches_gate(self):
        # The fused decode kernel must reproduce the reference: it folds the new
        # token's key into the rolling [max|min|sum] accumulator, and when the
        # request's block just filled (seq_len % block_size == 0) recovers the
        # block's max|min|avg from the freshly-updated reductions, compresses,
        # and RoPEs at the completing-token position (seq_len - 1).  Requests not
        # on a block boundary must NOT write a summary.
        from sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate import (
            AttnGate,
            apply_rotary_pos_emb_seer,
        )
        from sglang.srt.layers.attention.blocksparse.seer_attn.cache_kernels import (
            update_summary_cache_decode,
        )

        torch.manual_seed(0)
        dev, dt = "cuda", torch.bfloat16
        block_size, Hk, D, Hq = 64, 4, 128, 16
        GATE_D = D
        theta, max_pos = 1_000_000.0, 8192

        # Mix of boundary (seq_len % block == 0) and non-boundary requests.  The
        # boundary requests (128, 64) have pos_in_block == block_size-1, so the
        # in-kernel fold of the new (last) token completes their block.
        seq_lens = torch.tensor([128, 130, 64, 100], dtype=torch.int32, device=dev)
        bsz = seq_lens.shape[0]
        req_pool_indices = torch.tensor([3, 1, 0, 2], device=dev)

        max_blocks = 8
        max_reqs = 8
        req_to_summary = torch.arange(
            max_reqs * max_blocks, device=dev, dtype=torch.int64
        ).view(max_reqs, max_blocks)
        summary_pool = max_reqs * max_blocks + 1

        # Full raw block per request (the reference source of truth).  The
        # accumulator is seeded from the first block_size-1 tokens; the last
        # token is fed as the kernel's "new token" k, so the in-kernel fold
        # reconstructs the full-block [max|min|sum].
        full_blocks = torch.randn(max_reqs, block_size, Hk, D, device=dev, dtype=dt)
        head = full_blocks[:, : block_size - 1].float()  # first 63 tokens
        rolling = torch.cat(
            [head.amax(1), head.amin(1), head.sum(1)], dim=-1
        )  # [max_reqs, Hk, 3*D] fp32 (partial-block accumulator)
        # New token (the block_size-th), indexed in batch order.
        k_new = full_blocks[req_pool_indices, block_size - 1].contiguous()  # [bsz,Hk,D]

        gate = AttnGate(
            block_size=block_size,
            model_hidden_size=D,
            gate_hidden_size=GATE_D,
            num_k_head=Hk,
            num_q_head=Hq,
        ).to(dev)
        with torch.no_grad():
            gate.attngate_linear_k.weight.normal_(0, 0.05)
            gate.attngate_knorm.weight.uniform_(0.5, 1.5)
        gate = gate.to(dt)

        cos_tab, sin_tab, inv_freq = self._rope_tables(
            GATE_D, theta, max_pos, dev, dt
        )

        # Seed the summary pool with a sentinel so we can detect stray writes.
        summary_k = torch.full(
            (summary_pool, Hk, GATE_D), -7.0, device=dev, dtype=dt
        )
        update_summary_cache_decode(
            k_new,
            rolling,
            seq_lens,
            req_pool_indices,
            req_to_summary,
            summary_k,
            gate.attngate_linear_k.weight,
            gate.attngate_knorm.weight,
            inv_freq,
            block_size=block_size,
        )

        rpi = req_pool_indices.tolist()
        any_fill = False
        for i in range(bsz):
            sl = int(seq_lens[i])
            block_id = (sl - 1) // block_size
            slot = int(req_to_summary[rpi[i], block_id])
            if sl % block_size == 0:
                any_fill = True
                # Reference: pool the full block -> compress -> RoPE @ sl-1.
                kb = full_blocks[rpi[i]].unsqueeze(0)  # [1, block, Hk, D]
                comp = gate.compress_k(gate.pool_block_keys(kb))  # [1, Hk, GATE_D]
                pos = torch.tensor([sl - 1], device=dev)
                comp = apply_rotary_pos_emb_seer(
                    comp, cos_tab[pos].unsqueeze(1), sin_tab[pos].unsqueeze(1)
                )[0]
                diff = (summary_k[slot].float() - comp.float()).abs().max().item()
                self.assertLess(diff, 3e-2, f"req {i} summary diff {diff}")
            else:
                # Non-boundary request: its current-block slot must be untouched.
                self.assertTrue(
                    bool((summary_k[slot] == -7.0).all()),
                    f"req {i} wrote a summary on a non-boundary step",
                )
        self.assertTrue(any_fill, "test must exercise at least one block fill")


@unittest.skipUnless(_HAS_CUDA, "Decode block-index kernel requires CUDA")
class TestSeerAttnDecodeBlockIndex(CustomTestCase):
    """The seer decode block-selection path (gate-query RoPE hoisted into torch,
    then the shared dot-product top-k selector ``compute_active_block_ids`` with
    no forced bands) must select the same *previous* blocks as the reference
    PyTorch path (RoPE q at the current position -> gather over complete previous
    blocks -> einsum score -> top-block_budget).  The current/partial block is
    excluded from selection (appended downstream by position), so the reference
    scores only blocks ``[0, num_prev)``.  Softmax is intentionally absent: it is
    monotonic and does not change a top-k.  Covers a ragged batch with both
    complete and incomplete trailing blocks."""

    def _rope_tables(self, rope_dim, theta, max_pos, device, dtype):
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
                / rope_dim
            )
        )
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype), inv_freq

    def _rope_q_gate(self, q_gate, seq_lens, inv_freq):
        """Reference gate-query RoPE (NeoX, at the current token position
        ``seq_len - 1``) via the torch ``apply_rotary_pos_emb_seer`` path — the
        ground truth the fused ``rope_gate_query`` kernel must match."""
        from sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate import (
            apply_rotary_pos_emb_seer,
        )

        bsz = seq_lens.shape[0]
        pos = (seq_lens.to(torch.float32) - 1.0).view(bsz, 1)
        angle = pos * inv_freq.view(1, -1)
        emb = torch.cat((angle, angle), dim=-1)
        cos = emb.cos().unsqueeze(1).to(q_gate.dtype)
        sin = emb.sin().unsqueeze(1).to(q_gate.dtype)
        return apply_rotary_pos_emb_seer(q_gate, cos, sin).contiguous()

    def test_fused_rope_matches_torch(self):
        """The fused ``rope_gate_query`` Triton kernel must match the torch
        ``apply_rotary_pos_emb_seer`` reference (fp32 rotation), across a ragged
        batch of positions."""
        from sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate import (
            rope_gate_query,
        )

        torch.manual_seed(0)
        dev, dt = "cuda", torch.float32
        Hk, gate_dim = 4, 128
        theta, max_pos = 1_000_000.0, 8192
        seq_lens = torch.tensor([130, 100, 200, 64, 1, 2], dtype=torch.int32, device=dev)
        bsz = seq_lens.shape[0]
        q_gate = torch.randn(bsz, Hk, gate_dim, device=dev, dtype=dt)
        _, _, inv_freq = self._rope_tables(gate_dim, theta, max_pos, dev, dt)

        got = rope_gate_query(q_gate, seq_lens, inv_freq)
        ref = self._rope_q_gate(q_gate, seq_lens, inv_freq)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_block_index_matches_reference(self):
        from sglang.srt.layers.attention.blocksparse.common_index_kernels import (
            compute_active_block_ids,
        )
        from sglang.srt.layers.attention.blocksparse.seer_attn.attn_gate import (
            apply_rotary_pos_emb_seer,
            rope_gate_query,
        )

        torch.manual_seed(0)
        dev, dt = "cuda", torch.float32
        block_size, Hk, gate_dim = 64, 4, 128
        scale = 1.0 / (gate_dim**0.5)
        budget_blocks = 3  # token_budget // block_size
        theta, max_pos = 1_000_000.0, 8192

        # seq_lens with mixed complete (128) and incomplete (130, 100, 200, 64).
        seq_lens = torch.tensor([130, 100, 200, 64, 128], dtype=torch.int32, device=dev)
        bsz = seq_lens.shape[0]
        req_pool_indices = torch.tensor([4, 2, 0, 3, 1], device=dev)

        MAX_BLOCKS = 8
        max_reqs = 8
        # Distinct, valid summary slots per (req, block).  Slot 0 stays a zero
        # sentinel; assign 1..N.
        req_to_summary = (
            torch.arange(1, max_reqs * MAX_BLOCKS + 1, device=dev, dtype=torch.int32)
        ).view(max_reqs, MAX_BLOCKS)
        summary_pool = max_reqs * MAX_BLOCKS + 2
        summary_buf = torch.randn(summary_pool, Hk, gate_dim, device=dev, dtype=dt)
        summary_buf[0] = 0.0  # sentinel slot

        q_gate = torch.randn(bsz, Hk, gate_dim, device=dev, dtype=dt)
        cos_tab, sin_tab, inv_freq = self._rope_tables(
            gate_dim, theta, max_pos, dev, dt
        )

        # New path: hoist gate-query RoPE via the fused kernel (as the backend
        # does), then run the shared dot-product top-k selector with no forced
        # bands.
        q_gate_rope = rope_gate_query(q_gate, seq_lens, inv_freq)
        out = compute_active_block_ids(
            q_gate_rope,
            summary_buf,
            seq_lens,
            req_pool_indices,
            req_to_summary,
            block_size=block_size,
            topk=budget_blocks,
            num_init_blocks=0,
            num_local_blocks=0,
        )  # [bsz, Hk, block_budget] int32, -1 padded

        # --- reference: RoPE q @ (seq_len-1) -> gather over previous blocks ---
        # Only complete previous blocks [0, num_prev) are candidates; the current
        # (partial) block is excluded (handled downstream by position).
        num_prev = ((seq_lens - 1) // block_size).to(torch.long)  # [bsz]
        block_ar = torch.arange(MAX_BLOCKS, device=dev)
        valid = block_ar[None, :] < num_prev[:, None]
        slot = req_to_summary[req_pool_indices].long()
        slot = torch.where(valid, slot, torch.zeros_like(slot))
        k_blocks = summary_buf[slot].clone()  # [bsz, MAX_BLOCKS, Hk, gate_dim]
        # RoPE the gate query at each request's current position (seq_len - 1).
        pos = (seq_lens.long() - 1)
        q_rope = apply_rotary_pos_emb_seer(
            q_gate, cos_tab[pos].unsqueeze(1), sin_tab[pos].unsqueeze(1)
        )
        attn = torch.einsum("bkd,bskd->bks", q_rope, k_blocks) * scale
        valid_kh = valid[:, None, :].expand(bsz, Hk, MAX_BLOCKS)
        attn = attn.masked_fill(~valid_kh, float("-inf"))
        # top-block_budget over valid previous blocks (no softmax — monotonic, so
        # it would not change the top-k; no forced last block).
        if MAX_BLOCKS <= budget_blocks:
            mask = torch.ones_like(attn, dtype=torch.bool)
        else:
            _, topk_idx = torch.topk(attn, k=budget_blocks, dim=-1, sorted=False)
            mask = torch.zeros_like(attn, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
        mask = mask & valid_kh

        # Compare as sets of selected block ids per (req, head).
        for b in range(bsz):
            for h in range(Hk):
                got = sorted(x for x in out[b, h].tolist() if x >= 0)
                ref = sorted(torch.nonzero(mask[b, h]).flatten().tolist())
                self.assertEqual(got, ref, f"(req {b}, head {h}) mismatch")


class TestSeerAttnGateWeightSharding(CustomTestCase):
    """Gate-weight TP sharding (pure CPU, no CUDA / no checkpoint).

    The released ``attn_gate_weights.pth`` is full-head; under attention-TP the
    per-head projections must be sliced along the kv-head axis (dim 0) and the
    norm vectors replicated.  Concatenating every rank's slice along dim 0 must
    reconstruct the original full-head weight exactly (no overlap, no gap).
    """

    def test_linear_shards_partition_and_roundtrip(self):
        from sglang.srt.models.qwen3_seer import _shard_gate_weight

        torch.manual_seed(0)
        num_kv_heads, gqa, D, GATE_D = 8, 2, 128, 128
        tp_size = 4  # 8 kv heads / 4 ranks -> 2 kv heads each (partition regime)

        wq = torch.randn(num_kv_heads, gqa, D, GATE_D)
        wk = torch.randn(num_kv_heads, D * 3, GATE_D)

        for full, nm in ((wq, "model.layers.0.self_attn.attn_gate.attngate_linear_q.weight"),
                          (wk, "model.layers.0.self_attn.attn_gate.attngate_linear_k.weight")):
            shards = [
                _shard_gate_weight(nm, full, attn_tp_rank=r, attn_tp_size=tp_size)
                for r in range(tp_size)
            ]
            # Each rank gets an even, non-overlapping kv-head slab.
            local = num_kv_heads // tp_size
            for r, s in enumerate(shards):
                self.assertEqual(s.shape[0], local)
                self.assertTrue(
                    torch.equal(s, full[r * local : (r + 1) * local])
                )
            # Concatenation along dim 0 reconstructs the full weight.
            self.assertTrue(torch.equal(torch.cat(shards, dim=0), full))

    def test_norm_weights_replicated(self):
        from sglang.srt.models.qwen3_seer import _shard_gate_weight

        GATE_D, tp_size = 128, 4
        for nm in ("model.layers.0.self_attn.attn_gate.attngate_qnorm.weight",
                   "model.layers.0.self_attn.attn_gate.attngate_knorm.weight"):
            w = torch.randn(GATE_D)
            for r in range(tp_size):
                out = _shard_gate_weight(nm, w, attn_tp_rank=r, attn_tp_size=tp_size)
                self.assertTrue(torch.equal(out, w))  # full copy on every rank

    def test_tp1_is_noop(self):
        from sglang.srt.models.qwen3_seer import _shard_gate_weight

        w = torch.randn(8, 128, 128)
        nm = "model.layers.0.self_attn.attn_gate.attngate_linear_k.weight"
        self.assertTrue(
            torch.equal(_shard_gate_weight(nm, w, attn_tp_rank=0, attn_tp_size=1), w)
        )


if __name__ == "__main__":
    unittest.main()
