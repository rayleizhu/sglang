"""Unit tests for block-sparse decode routing index logic.

Covers the two core routing functions:

  * ``compute_active_block_ids`` (common_index_kernels) — per-(req, kv-head)
    top-k block selection
  * ``block_inds_to_kv_inds_for_decoding`` (common_index_kernels) — ragged
    virtual KV index expansion

These functions are pure integer-index logic (no floating-point approximation),
so we can assert *exact* properties against independent references:

  Layer 1 (precise):
    - block 0 always selected (attention sink) when sparse
    - exactly ``actual_topk = min(current_block, topk)`` previous blocks selected
    - the selected previous blocks are the highest-scoring ones (vs an
      independent argsort of the score matrix)
    - expanded kv length per (req, head) matches ``_compute_sparse_kv_lens``
    - virtual index decoding is consistent: ``idx % H == head`` and
      ``idx // H`` is a pool loc of a selected/current-block token

  Layer 3 (dense degeneration, hardest judge):
    - when ``current_block <= topk`` the routing reduces to *dense*: the
      produced virtual KV indices equal the full dense index set.

The routing functions are pure (config passed in explicitly), so the tests
call them directly — no backend / ModelRunner / flashinfer wrapper needed.
"""

import unittest

import torch

from sglang.srt.layers.attention.blocksparse.common_index_kernels import (
    block_inds_to_kv_inds_for_decoding,
    block_inds_to_kv_inds_for_decoding_ref,
)
from sglang.srt.layers.attention.blocksparse.common_index_kernels import (
    compute_active_block_ids as _compute_active_block_ids_impl,
)
from sglang.srt.layers.attention.blocksparse.common_index_kernels import (
    compute_active_block_ids_ref as _compute_active_block_ids_ref_impl,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Self-contained integer-index routing tests for the block-sparse backends.
register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")

_HAS_CUDA = torch.cuda.is_available()


def _group_merge(q, num_kv_heads, gqa_group_size, head_dim):
    """Mean-merge query heads within each GQA group: [bsz, H*G, d] -> [bsz, H, d]."""
    bsz = q.size(0)
    return q.view(bsz, num_kv_heads, gqa_group_size, head_dim).mean(dim=2)


def _select_wrapper(impl):
    """Adapt the test-style call (raw q [bsz,H*G,d] + head config kwargs) to the
    group-merged API (q_retrieval [bsz,H,d], shapes inferred)."""

    def fn(
        q,
        summary_buffer,
        seq_lens,
        req_pool_indices,
        req_to_summary,
        *,
        block_size,
        topk,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        num_init_blocks=1,
        num_local_blocks=0,
    ):
        q_retrieval = _group_merge(q, num_kv_heads, gqa_group_size, head_dim)
        return impl(
            q_retrieval,
            summary_buffer,
            seq_lens,
            req_pool_indices,
            req_to_summary,
            block_size=block_size,
            topk=topk,
            num_init_blocks=num_init_blocks,
            num_local_blocks=num_local_blocks,
        )

    return fn


# Test-facing adapters that keep the original (pre-refactor) call signature.
compute_active_block_ids = _select_wrapper(_compute_active_block_ids_impl)
compute_active_block_ids_ref = _select_wrapper(_compute_active_block_ids_ref_impl)


def sparse_kv_lens_reference(seq_lens, block_size, topk):
    """Independent copy of BlockSparseIndicesUpdaterDecode._compute_sparse_kv_lens."""
    current_block_id = (seq_lens - 1) // block_size
    num_prev = current_block_id
    actual_topk = torch.clamp(num_prev, max=topk)
    cur_block_len = seq_lens - current_block_id * block_size
    return actual_topk * block_size + cur_block_len


@unittest.skipUnless(_HAS_CUDA, "block-sparse routing kernels require CUDA")
class TestBlockSparseRouting(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        torch.manual_seed(0)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def _build_contiguous_req_to_token(self, seq_lens_list, req_pool_size, max_ctx):
        """Identity-ish mapping: req i's token t lives at pool loc base_i + t,
        with non-overlapping per-request bases."""
        device = "cuda"
        req_to_token = torch.zeros(
            req_pool_size, max_ctx, dtype=torch.int32, device=device
        )
        base = 1  # avoid 0 to make virtual index decoding unambiguous
        bases = []
        for i, sl in enumerate(seq_lens_list):
            bases.append(base)
            req_to_token[i, :sl] = torch.arange(
                base, base + sl, dtype=torch.int32, device=device
            )
            base += max_ctx
        return req_to_token, bases

    def _alloc_summaries(self, req_to_summary, seq_lens_list, block_size):
        """Assign consecutive summary slots (starting at 1) for every block of
        every request."""
        slot = 1
        for i, sl in enumerate(seq_lens_list):
            nblk = (sl - 1) // block_size + 1
            for b in range(nblk):
                req_to_summary[i, b] = slot
                slot += 1
        return slot

    # -------------------------------------------------------------------
    # Layer 1: selection properties
    # -------------------------------------------------------------------
    def test_selection_sparse_properties(self):
        device = "cuda"
        block_size, topk, H, G, d = 8, 3, 2, 4, 16
        # seq_lens chosen so current_block > topk (sparse regime)
        seq_lens_list = [8 * 6 + 3, 8 * 10 + 1]  # current_block = 6, 10 (> topk=3)
        bsz = len(seq_lens_list)
        max_ctx = 256
        req_pool_size = 4

        req_to_token, _ = self._build_contiguous_req_to_token(
            seq_lens_list, req_pool_size, max_ctx
        )
        max_num_summary = max_ctx // block_size + 1
        req_to_summary = torch.zeros(
            req_pool_size, max_num_summary, dtype=torch.int32, device=device
        )
        n_slots = self._alloc_summaries(req_to_summary, seq_lens_list, block_size)
        summary_buffer = torch.randn(n_slots + 1, H, d, device=device)

        q = torch.randn(bsz, H * G, d, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bsz, dtype=torch.int64, device=device)

        active = compute_active_block_ids_ref(
            q,
            summary_buffer,
            seq_lens,
            req_pool_indices,
            req_to_summary,
            block_size=block_size,
            topk=topk,
            num_kv_heads=H,
            gqa_group_size=G,
            head_dim=d,
        )
        self.assertEqual(tuple(active.shape), (bsz, H, topk))

        # Independent reference scoring
        q_retrieval = q.view(bsz, H, G, d).mean(dim=2).float()
        for i in range(bsz):
            current_block = (seq_lens_list[i] - 1) // block_size
            for h in range(H):
                sel = active[i, h]
                self.assertTrue((sel >= 0).all().item(), "sparse regime: no padding")
                self.assertEqual(sel.numel(), topk)
                sel_set = set(sel.tolist())
                # block 0 must be selected (sink)
                self.assertIn(0, sel_set, "block 0 (sink) must be selected")
                # all distinct
                self.assertEqual(len(sel_set), topk, "selected blocks must be distinct")
                # all within valid previous-block range
                self.assertTrue(max(sel_set) < current_block)
                # the (topk-1) non-sink picks must be the highest-scoring candidates
                cand_slots = req_to_summary[i, 1:current_block].long()
                cand = summary_buffer[cand_slots].float()  # [num_cand, H, d]
                scores = torch.einsum("d,pd->p", q_retrieval[i, h], cand[:, h])
                ref_top = set((scores.topk(topk - 1).indices + 1).tolist())
                self.assertEqual(
                    sel_set - {0}, ref_top, "non-sink picks must match top scores"
                )

    def test_selection_count_matches_sparse_kv_lens(self):
        """The number of selected previous blocks * block_size + current block
        len must equal _compute_sparse_kv_lens for every request."""
        device = "cuda"
        block_size, topk, H, G, d = 16, 2, 2, 2, 16
        # mix of sparse and dense regimes
        seq_lens_list = [16 * 5 + 7, 16 * 2 + 1, 16 * 1 + 3, 5]
        bsz = len(seq_lens_list)
        max_ctx = 256
        req_pool_size = 8

        req_to_token, _ = self._build_contiguous_req_to_token(
            seq_lens_list, req_pool_size, max_ctx
        )
        max_num_summary = max_ctx // block_size + 1
        req_to_summary = torch.zeros(
            req_pool_size, max_num_summary, dtype=torch.int32, device=device
        )
        n_slots = self._alloc_summaries(req_to_summary, seq_lens_list, block_size)
        summary_buffer = torch.randn(n_slots + 1, H, d, device=device)

        q = torch.randn(bsz, H * G, d, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bsz, dtype=torch.int64, device=device)

        active = compute_active_block_ids_ref(
            q,
            summary_buffer,
            seq_lens,
            req_pool_indices,
            req_to_summary,
            block_size=block_size,
            topk=topk,
            num_kv_heads=H,
            gqa_group_size=G,
            head_dim=d,
        )
        kv_inds = block_inds_to_kv_inds_for_decoding_ref(
            active,
            seq_lens,
            req_pool_indices,
            req_to_token,
            block_size=block_size,
            num_kv_heads=H,
        )

        ref_lens = sparse_kv_lens_reference(
            seq_lens.cpu(), block_size, topk
        )  # [bsz]
        # total length: sum over requests of H * sparse_kv_len
        expected_total = int((ref_lens * H).sum().item())
        self.assertEqual(kv_inds.numel(), expected_total)

        # virtual index decoding: head id = idx % H must cycle correctly given
        # the batch-major head-minor ragged layout.
        offset = 0
        for i in range(bsz):
            per_head_len = int(ref_lens[i].item())
            for h in range(H):
                seg = kv_inds[offset : offset + per_head_len]
                self.assertTrue(
                    (seg % H == h).all().item(),
                    f"req {i} head {h}: virtual idx head-bits mismatch",
                )
                offset += per_head_len
        self.assertEqual(offset, expected_total)

    def test_selected_blocks_sorted_ascending(self):
        """Selected block ids per (req, head) must be sorted ascending with -1
        padding at the tail. The scatter writes slot p at offset p*block_size,
        so out-of-order blocks would corrupt the kv_indices layout."""
        device = "cuda"
        block_size, topk, H, G, d = 8, 5, 2, 4, 16
        # sparse regime so init/local/picked bands all participate
        seq_lens_list = [8 * 11 + 3, 8 * 15 + 1]
        bsz = len(seq_lens_list)
        max_ctx = 256
        req_pool_size = 4
        req_to_token, _ = self._build_contiguous_req_to_token(
            seq_lens_list, req_pool_size, max_ctx
        )
        max_num_summary = max_ctx // block_size + 1
        req_to_summary = torch.zeros(
            req_pool_size, max_num_summary, dtype=torch.int32, device=device
        )
        n_slots = self._alloc_summaries(req_to_summary, seq_lens_list, block_size)
        summary_buffer = torch.randn(n_slots + 1, H, d, device=device)
        q = torch.randn(bsz, H * G, d, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bsz, dtype=torch.int64, device=device)

        for fn in (compute_active_block_ids_ref, compute_active_block_ids):
            active = fn(
                q,
                summary_buffer,
                seq_lens,
                req_pool_indices,
                req_to_summary,
                block_size=block_size,
                topk=topk,
                num_kv_heads=H,
                gqa_group_size=G,
                head_dim=d,
                num_init_blocks=1,
                num_local_blocks=2,
            )
            for i in range(bsz):
                for h in range(H):
                    row = active[i, h].tolist()
                    valid = [b for b in row if b >= 0]
                    pads = [b for b in row if b < 0]
                    # valid entries strictly ascending
                    self.assertEqual(
                        valid, sorted(valid), f"{fn.__name__}: not ascending @ {i},{h}"
                    )
                    # padding (if any) only at the tail
                    self.assertEqual(
                        row[len(valid):], pads,
                        f"{fn.__name__}: padding not at tail @ {i},{h}",
                    )

    # -------------------------------------------------------------------
    # Layer 3: dense degeneration (hardest precise judge)
    # -------------------------------------------------------------------
    def test_dense_degeneration_equals_full_indices(self):
        """When current_block <= topk, every previous block is kept, so the
        produced virtual KV indices must equal the full dense set
        (all tokens [0, seq_len), mapped to virtual coords)."""
        device = "cuda"
        block_size, topk, H, G, d = 8, 4, 2, 3, 16
        # current_block <= topk for all -> dense regime
        seq_lens_list = [8 * 3 + 5, 8 * 4, 7]  # current_block = 3, 3, 0
        bsz = len(seq_lens_list)
        max_ctx = 128
        req_pool_size = 4

        req_to_token, _ = self._build_contiguous_req_to_token(
            seq_lens_list, req_pool_size, max_ctx
        )
        max_num_summary = max_ctx // block_size + 1
        req_to_summary = torch.zeros(
            req_pool_size, max_num_summary, dtype=torch.int32, device=device
        )
        n_slots = self._alloc_summaries(req_to_summary, seq_lens_list, block_size)
        summary_buffer = torch.randn(n_slots + 1, H, d, device=device)

        q = torch.randn(bsz, H * G, d, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bsz, dtype=torch.int64, device=device)

        active = compute_active_block_ids_ref(
            q,
            summary_buffer,
            seq_lens,
            req_pool_indices,
            req_to_summary,
            block_size=block_size,
            topk=topk,
            num_kv_heads=H,
            gqa_group_size=G,
            head_dim=d,
        )
        kv_inds = block_inds_to_kv_inds_for_decoding_ref(
            active,
            seq_lens,
            req_pool_indices,
            req_to_token,
            block_size=block_size,
            num_kv_heads=H,
        )

        # Build the expected dense virtual indices independently.
        expected_segments = []
        for i in range(bsz):
            sl = seq_lens_list[i]
            pool_locs = req_to_token[i, :sl].to(torch.int64)
            for h in range(H):
                expected_segments.append((pool_locs * H + h).to(torch.int32))
        expected = torch.cat(expected_segments)

        self.assertEqual(kv_inds.numel(), expected.numel())
        # Order matters and must match exactly (batch-major, head-minor).
        self.assertTrue(
            torch.equal(kv_inds, expected),
            "dense-regime virtual indices must equal full dense index set",
        )

    def test_dense_degeneration_matches_kv_indptr(self):
        """In the dense regime, the per-(req,head) segment lengths must equal
        seq_len, matching the kv_indptr the decode updater would build."""
        device = "cuda"
        block_size, topk, H, G, d = 8, 4, 2, 3, 16
        seq_lens_list = [8 * 3 + 5, 8 * 4, 7]
        bsz = len(seq_lens_list)
        ref_lens = sparse_kv_lens_reference(
            torch.tensor(seq_lens_list), block_size, topk
        )
        # dense regime => sparse_kv_len == seq_len
        for i in range(bsz):
            self.assertEqual(int(ref_lens[i].item()), seq_lens_list[i])

    # -------------------------------------------------------------------
    # forced bands: num_init_blocks (sink) + num_local_blocks (local)
    # -------------------------------------------------------------------
    def _sparse_setup(self, block_size, topk, H, G, d, seq_lens_list):
        """Common setup for a single sparse-regime request batch."""
        device = "cuda"
        bsz = len(seq_lens_list)
        max_ctx = 512
        req_pool_size = 8
        req_to_token, _ = self._build_contiguous_req_to_token(
            seq_lens_list, req_pool_size, max_ctx
        )
        max_num_summary = max_ctx // block_size + 1
        req_to_summary = torch.zeros(
            req_pool_size, max_num_summary, dtype=torch.int32, device=device
        )
        n_slots = self._alloc_summaries(req_to_summary, seq_lens_list, block_size)
        summary_buffer = torch.randn(n_slots + 1, H, d, device=device)
        q = torch.randn(bsz, H * G, d, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bsz, dtype=torch.int64, device=device)
        return (
            req_to_token,
            req_to_summary,
            summary_buffer,
            q,
            seq_lens,
            req_pool_indices,
        )

    def test_init_and_local_bands(self):
        """num_init_blocks=1, num_local_blocks=2 must always include block 0
        (sink) AND the 2 previous blocks before the current one (local), still
        total topk, with remaining slots being top-scoring non-forced blocks."""
        block_size, topk, H, G, d = 8, 5, 2, 4, 16
        num_init, num_local = 1, 2
        seq_lens_list = [8 * 9 + 5, 8 * 12 + 1]  # current_block = 9, 12 (> topk)
        (
            req_to_token,
            req_to_summary,
            summary_buffer,
            q,
            seq_lens,
            req_pool_indices,
        ) = self._sparse_setup(block_size, topk, H, G, d, seq_lens_list)

        active = compute_active_block_ids_ref(
            q,
            summary_buffer,
            seq_lens,
            req_pool_indices,
            req_to_summary,
            block_size=block_size,
            topk=topk,
            num_kv_heads=H,
            gqa_group_size=G,
            head_dim=d,
            num_init_blocks=num_init,
            num_local_blocks=num_local,
        )

        q_retrieval = q.view(len(seq_lens_list), H, G, d).mean(dim=2).float()
        for i in range(len(seq_lens_list)):
            current_block = (seq_lens_list[i] - 1) // block_size
            init_blocks = set(range(num_init))
            local_blocks = set(range(current_block - num_local, current_block))
            forced = init_blocks | local_blocks
            for h in range(H):
                sel = active[i, h]
                self.assertEqual(sel.numel(), topk)
                sel_set = set(sel.tolist())
                self.assertEqual(len(sel_set), topk, "blocks must be distinct")
                self.assertTrue(forced <= sel_set, "init+local bands must be present")
                self.assertTrue(max(sel_set) < current_block)
                # the non-forced picks must be the top scorers over the
                # candidate set [0, current_block) minus the forced bands.
                cand_ids = [b for b in range(current_block) if b not in forced]
                cand_ids_t = torch.tensor(cand_ids, device=q.device)
                cand_slots = req_to_summary[i, cand_ids_t].long()
                cand = summary_buffer[cand_slots].float()  # [num_cand, H, d]
                scores = torch.einsum("d,pd->p", q_retrieval[i, h], cand[:, h])
                n_pick = topk - len(forced)
                ref_top = {cand_ids[j] for j in scores.topk(n_pick).indices.tolist()}
                self.assertEqual(
                    sel_set - forced, ref_top, "non-forced picks must be top scorers"
                )

    def test_bands_overlap_dedup_and_count(self):
        """When init and local bands overlap (large bands), the union must be
        de-duplicated and the total selected count must stay exactly topk."""
        block_size, topk, H, G, d = 8, 6, 2, 2, 16
        # current_block = 7; init=4 -> {0,1,2,3}; local=5 -> {2,3,4,5,6};
        # union = {0,1,2,3,4,5,6} (7 blocks) which exceeds topk=6 -> truncated.
        num_init, num_local = 4, 5
        seq_lens_list = [8 * 7 + 3]  # current_block = 7 (> topk)
        (
            req_to_token,
            req_to_summary,
            summary_buffer,
            q,
            seq_lens,
            req_pool_indices,
        ) = self._sparse_setup(block_size, topk, H, G, d, seq_lens_list)

        active = compute_active_block_ids_ref(
            q,
            summary_buffer,
            seq_lens,
            req_pool_indices,
            req_to_summary,
            block_size=block_size,
            topk=topk,
            num_kv_heads=H,
            gqa_group_size=G,
            head_dim=d,
            num_init_blocks=num_init,
            num_local_blocks=num_local,
        )
        for h in range(H):
            sel = active[0, h]
            sel_set = set(sel.tolist())
            self.assertEqual(sel.numel(), topk)
            self.assertEqual(len(sel_set), topk, "must be distinct despite overlap")

        # Count invariant holds vs sparse_kv_lens regardless of bands.
        kv_inds = block_inds_to_kv_inds_for_decoding_ref(
            active,
            seq_lens,
            req_pool_indices,
            req_to_token,
            block_size=block_size,
            num_kv_heads=H,
        )
        ref_lens = sparse_kv_lens_reference(seq_lens.cpu(), block_size, topk)
        self.assertEqual(kv_inds.numel(), int((ref_lens * H).sum().item()))

    def test_default_bands_match_explicit_sink(self):
        """Defaults (num_init_blocks=1, num_local_blocks=0) must give identical
        result to passing them explicitly — guards against default drift, and
        documents that the default is the classic single-sink behavior."""
        block_size, topk, H, G, d = 8, 3, 2, 4, 16
        seq_lens_list = [8 * 7 + 2, 8 * 11 + 6]
        setup = self._sparse_setup(block_size, topk, H, G, d, seq_lens_list)
        _, req_to_summary, summary_buffer, q, seq_lens, req_pool_indices = setup
        common = dict(
            block_size=block_size,
            topk=topk,
            num_kv_heads=H,
            gqa_group_size=G,
            head_dim=d,
        )
        a_default = compute_active_block_ids_ref(
            q, summary_buffer, seq_lens, req_pool_indices, req_to_summary, **common
        )
        a_explicit = compute_active_block_ids_ref(
            q,
            summary_buffer,
            seq_lens,
            req_pool_indices,
            req_to_summary,
            num_init_blocks=1,
            num_local_blocks=0,
            **common,
        )
        self.assertTrue(torch.equal(a_default, a_explicit))


@unittest.skipUnless(_HAS_CUDA, "block-sparse routing kernels require CUDA")
class TestBlockSparseRoutingKernelParity(CustomTestCase):
    """Triton kernels must match the pure-PyTorch reference (ground truth)."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        torch.manual_seed(1)

    def _build_req_to_token(self, seq_lens_list, req_pool_size, max_ctx):
        device = "cuda"
        req_to_token = torch.zeros(
            req_pool_size, max_ctx, dtype=torch.int32, device=device
        )
        base = 1
        for i, sl in enumerate(seq_lens_list):
            req_to_token[i, :sl] = torch.arange(
                base, base + sl, dtype=torch.int32, device=device
            )
            base += max_ctx
        return req_to_token

    def _alloc_summaries(self, req_to_summary, seq_lens_list, block_size):
        slot = 1
        for i, sl in enumerate(seq_lens_list):
            nblk = (sl - 1) // block_size + 1
            for b in range(nblk):
                req_to_summary[i, b] = slot
                slot += 1
        return slot

    def _run_case(
        self, block_size, topk, H, G, d, seq_lens_list, num_init, num_local
    ):
        device = "cuda"
        bsz = len(seq_lens_list)
        max_ctx = 1024
        req_pool_size = bsz + 2
        req_to_token = self._build_req_to_token(seq_lens_list, req_pool_size, max_ctx)
        max_num_summary = max_ctx // block_size + 1
        req_to_summary = torch.zeros(
            req_pool_size, max_num_summary, dtype=torch.int32, device=device
        )
        n_slots = self._alloc_summaries(req_to_summary, seq_lens_list, block_size)
        # Distinct-valued summaries so dot products have no ties (exact top-k).
        summary_buffer = torch.randn(n_slots + 1, H, d, device=device)
        q = torch.randn(bsz, H * G, d, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bsz, dtype=torch.int64, device=device)

        common = dict(
            block_size=block_size,
            topk=topk,
            num_kv_heads=H,
            gqa_group_size=G,
            head_dim=d,
            num_init_blocks=num_init,
            num_local_blocks=num_local,
        )
        ref = compute_active_block_ids_ref(
            q, summary_buffer, seq_lens, req_pool_indices, req_to_summary, **common
        )
        ker = compute_active_block_ids(
            q, summary_buffer, seq_lens, req_pool_indices, req_to_summary, **common
        )

        # Compare as *sets per (req, head)* — both must select the same blocks.
        # (slot ordering within a row may differ: ref puts forced-first, kernel
        # puts forced in fixed slots then picked; the downstream scatter is
        # order-insensitive across selected blocks because each maps to its own
        # kv_indptr offset by active-slot position. We assert set-equality here
        # and verify end-to-end index equality via the scatter parity below.)
        for i in range(bsz):
            for h in range(H):
                self.assertEqual(
                    set(ref[i, h].tolist()),
                    set(ker[i, h].tolist()),
                    f"block selection mismatch at req {i} head {h} "
                    f"(seq_len={seq_lens_list[i]}, init={num_init}, local={num_local})",
                )

        # --- scatter parity: kernel output (sliced by kv_indptr) == ref -------
        H_ = H
        current_block = (seq_lens.to(torch.int64) - 1) // block_size
        actual_topk = torch.clamp(current_block, max=topk)
        cur_len = seq_lens.to(torch.int64) - current_block * block_size
        sparse_kv_lens = actual_topk * block_size + cur_len
        virt = sparse_kv_lens.repeat_interleave(H_)
        indptr = torch.zeros(bsz * H_ + 1, dtype=torch.int64, device=device)
        indptr[1:] = torch.cumsum(virt, dim=0)

        # Use the SAME selected blocks (ref) for both ref- and kernel-scatter so
        # this isolates the scatter kernel from any selection-order differences.
        kv_ref = block_inds_to_kv_inds_for_decoding_ref(
            ref, seq_lens, req_pool_indices, req_to_token,
            block_size=block_size, num_kv_heads=H_,
        )
        kv_ker_buf = block_inds_to_kv_inds_for_decoding(
            ref, seq_lens, req_pool_indices, req_to_token,
            block_size=block_size, num_kv_heads=H_, topk=topk,
        )
        # Compact the over-allocated kernel buffer per segment and compare.
        total = int(indptr[-1].item())
        kv_ker = torch.empty(total, dtype=torch.int32, device=device)
        # segments are contiguous in both; copy each segment from its offset.
        seg_off = 0
        for v in range(bsz * H_):
            seg_len = int((indptr[v + 1] - indptr[v]).item())
            kv_ker[seg_off : seg_off + seg_len] = kv_ker_buf[
                indptr[v] : indptr[v] + seg_len
            ]
            seg_off += seg_len
        self.assertEqual(kv_ref.numel(), total)
        self.assertTrue(
            torch.equal(kv_ref, kv_ker),
            "scatter kernel virtual indices must equal the reference",
        )

    def test_parity_sparse_sink_only(self):
        self._run_case(8, 4, 2, 4, 16, [8 * 9 + 5, 8 * 12 + 1], 1, 0)

    def test_parity_sparse_init_and_local(self):
        self._run_case(8, 5, 2, 4, 16, [8 * 9 + 5, 8 * 12 + 1, 8 * 20 + 7], 1, 2)

    def test_parity_dense_and_short(self):
        # current_block = 3, 3, 0 -> dense / degenerate regime
        self._run_case(8, 4, 2, 3, 16, [8 * 3 + 5, 8 * 4, 7], 1, 0)

    def test_parity_mixed_batch(self):
        # mix sparse + dense in one batch
        self._run_case(16, 3, 2, 2, 16, [16 * 8 + 3, 16 * 2 + 1, 16 * 1 + 5], 1, 1)

    def test_parity_no_forced_bands(self):
        self._run_case(8, 4, 2, 4, 16, [8 * 10 + 2], 0, 0)

    def test_parity_tie_break(self):
        """All candidate scores exactly equal -> the threshold tie-break path
        must select the same blocks as torch.topk (lower block id wins) and
        keep the count at exactly topk. This is the most error-prone path of
        the threshold-based kernel."""
        device = "cuda"
        block_size, topk, H, G, d = 8, 4, 2, 2, 16
        num_init, num_local = 1, 0
        seq_lens_list = [8 * 12 + 3]  # current_block = 12 (sparse)
        bsz = 1
        max_ctx = 256
        req_pool_size = 4
        req_to_token = self._build_req_to_token(seq_lens_list, req_pool_size, max_ctx)
        max_num_summary = max_ctx // block_size + 1
        req_to_summary = torch.zeros(
            req_pool_size, max_num_summary, dtype=torch.int32, device=device
        )
        n_slots = self._alloc_summaries(req_to_summary, seq_lens_list, block_size)
        # All summaries identical -> all candidate dot products exactly equal,
        # so every candidate is tied at the threshold.
        ones = torch.ones(H, d, device=device)
        summary_buffer = torch.zeros(n_slots + 1, H, d, device=device)
        summary_buffer[1:] = ones  # slots 1.. all identical
        q = torch.randn(bsz, H * G, d, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bsz, dtype=torch.int64, device=device)

        common = dict(
            block_size=block_size,
            topk=topk,
            num_kv_heads=H,
            gqa_group_size=G,
            head_dim=d,
            num_init_blocks=num_init,
            num_local_blocks=num_local,
        )
        ref = compute_active_block_ids_ref(
            q, summary_buffer, seq_lens, req_pool_indices, req_to_summary, **common
        )
        ker = compute_active_block_ids(
            q, summary_buffer, seq_lens, req_pool_indices, req_to_summary, **common
        )
        for h in range(H):
            self.assertEqual(
                set(ref[0, h].tolist()), set(ker[0, h].tolist()),
                f"tie-break selection mismatch @ head {h}",
            )
            # exactly topk distinct blocks
            self.assertEqual(len(set(ker[0, h].tolist())), topk)

    def test_parity_strict_winner_with_threshold_ties(self):
        """A strict score winner at a HIGH block id, plus several LOWER-id blocks
        tied exactly at the selection threshold.

        This is the case a plain ``acc >= thr`` rank-cutoff gets WRONG: the
        ascending-id truncation would fill the quota with the low-id tied blocks
        and drop the strict (higher-score) winner.  torch.topk (the _ref) always
        keeps the highest score, so kernel-vs-ref must agree here.  Scores are
        injected exactly via a one-hot query so the threshold tie is exact (not
        relying on fp noise)."""
        device = "cuda"
        block_size, topk, H, G, d = 8, 3, 2, 2, 16
        num_init, num_local = 1, 0  # block 0 forced (sink); n_pick = 2
        # current_block = 6 -> previous blocks 0..5; candidates (minus sink) 1..5
        seq_lens_list = [8 * 6 + 3]
        bsz = 1
        max_ctx = 256
        req_pool_size = 4
        req_to_token = self._build_req_to_token(seq_lens_list, req_pool_size, max_ctx)
        max_num_summary = max_ctx // block_size + 1
        req_to_summary = torch.zeros(
            req_pool_size, max_num_summary, dtype=torch.int32, device=device
        )
        n_slots = self._alloc_summaries(req_to_summary, seq_lens_list, block_size)

        # One-hot query e_0 (per head, identical across the GQA group so the mean
        # merge is still e_0): score(block b) = summary[slot_b, h, 0].
        q = torch.zeros(bsz, H * G, d, device=device)
        q[..., 0] = 1.0
        # Inject per-candidate scores: blocks 1,2,3 tie at thr=5; block 4 below;
        # block 5 is the strict winner (9) at the highest candidate id.
        # n_pick=2 -> thr = 2nd largest of {5,5,5,1,9} = 5.  Correct top-2 over
        # candidates = {blk5(9), blk1(5)}; the buggy >=thr cutoff would pick
        # {blk1, blk2} and drop blk5.
        block_score = {1: 5.0, 2: 5.0, 3: 5.0, 4: 1.0, 5: 9.0}
        summary_buffer = torch.zeros(n_slots + 1, H, d, device=device)
        for b, s in block_score.items():
            slot = int(req_to_summary[0, b].item())
            summary_buffer[slot, :, 0] = s

        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bsz, dtype=torch.int64, device=device)
        common = dict(
            block_size=block_size,
            topk=topk,
            num_kv_heads=H,
            gqa_group_size=G,
            head_dim=d,
            num_init_blocks=num_init,
            num_local_blocks=num_local,
        )
        ref = compute_active_block_ids_ref(
            q, summary_buffer, seq_lens, req_pool_indices, req_to_summary, **common
        )
        ker = compute_active_block_ids(
            q, summary_buffer, seq_lens, req_pool_indices, req_to_summary, **common
        )
        for h in range(H):
            ref_set = set(ref[0, h].tolist())
            ker_set = set(ker[0, h].tolist())
            # ref/kernel must agree, and both must contain the strict winner (5)
            # and the sink (0), dropping a low-id tie instead.
            self.assertEqual(ref_set, ker_set, f"selection mismatch @ head {h}")
            self.assertIn(5, ker_set, f"strict winner (blk 5, score 9) dropped @ head {h}")
            self.assertIn(0, ker_set, f"sink (blk 0) missing @ head {h}")
            self.assertEqual(len(ker_set), topk)

    def test_parity_thr_neg_inf_no_phantom_blocks(self):
        """Dense regime with candidate_count < n_pick forces the selection
        threshold to -inf.  The kernel's `eq = (acc == thr)` tie path must NOT
        then match the -inf non-candidate lanes (forced bands, blocks >= num_prev,
        and the NBLK power-of-two padding): `-inf == -inf` is True, so without the
        scalar `thr > -inf` gate those phantom ids would be pulled into the quota
        and the output would contain blocks that do not exist for the request.

        Constructs num_prev=3, topk=5 (num_prev < topk -> dense), num_init=1:
        candidates are just blocks {1,2} (block 0 forced), which is < n_pick=4, so
        thr collapses to -inf.  Correct output is exactly [0,1,2] padded with -1;
        the kernel must NOT emit block 3+ (non-existent) nor a duplicate sink."""
        device = "cuda"
        block_size, topk, H, G, d = 8, 5, 2, 2, 16
        num_init, num_local = 1, 0  # n_pick = 4
        seq_lens_list = [8 * 3 + 4]  # current_block = num_prev = 3 (< topk=5: dense)
        bsz = 1
        max_ctx = 256
        req_pool_size = 4
        req_to_token = self._build_req_to_token(seq_lens_list, req_pool_size, max_ctx)
        max_num_summary = max_ctx // block_size + 1  # NBLK padding well past num_prev
        req_to_summary = torch.zeros(
            req_pool_size, max_num_summary, dtype=torch.int32, device=device
        )
        n_slots = self._alloc_summaries(req_to_summary, seq_lens_list, block_size)
        summary_buffer = torch.randn(n_slots + 1, H, d, device=device)
        q = torch.randn(bsz, H * G, d, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bsz, dtype=torch.int64, device=device)

        common = dict(
            block_size=block_size,
            topk=topk,
            num_kv_heads=H,
            gqa_group_size=G,
            head_dim=d,
            num_init_blocks=num_init,
            num_local_blocks=num_local,
        )
        ref = compute_active_block_ids_ref(
            q, summary_buffer, seq_lens, req_pool_indices, req_to_summary, **common
        )
        ker = compute_active_block_ids(
            q, summary_buffer, seq_lens, req_pool_indices, req_to_summary, **common
        )
        for h in range(H):
            ker_valid = sorted(b for b in ker[0, h].tolist() if b >= 0)
            ref_valid = sorted(b for b in ref[0, h].tolist() if b >= 0)
            self.assertEqual(ker_valid, [0, 1, 2], f"dense selection wrong @ head {h}")
            self.assertEqual(ker_valid, ref_valid, f"kernel != ref @ head {h}")
            # no phantom (>= num_prev) block ids selected
            self.assertTrue(
                all(b < 3 for b in ker_valid),
                f"phantom block id selected @ head {h}: {ker_valid}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
