from __future__ import annotations

"""Representation-agnostic decode-time block selection + KV-index expansion for
block-sparse backends.

Two stages, both shared by every block-sparse backend (MoBA's rolling-mean
routing and SeerAttention-R's learned-gate routing):

* :func:`compute_active_block_ids` — per-(request, kv-head) block selection by
  query-summary similarity (top-``topk``), keeping optional forced ``num_init``
  leading sink blocks and ``num_local`` trailing local blocks.  The scoring is a
  plain ``q · summary`` dot product, so it is representation-agnostic given the
  caller's per-(kv-head) query and the cached per-block summaries: MoBA passes a
  GQA-group-merged raw query; SeerAttention-R passes the gate query after its
  Q-projection **and RoPE** (RoPE is the gate's representation choice, applied by
  the caller before this scoring — see the seer backend).  Produces the
  ``[bsz, num_kv_heads, topk]`` int32 (-1 padded, ascending) contract.

* :func:`block_inds_to_kv_inds_for_decoding` — expands the selected *logical*
  block ids (plus the current partial block, appended unconditionally by
  position) into ragged *virtual* KV indices in batch-major, head-minor order,
  matching the ``kv_indptr`` built via ``repeat_interleave(num_kv_heads)``.  This
  stage depends ONLY on ``active_block_ids`` + ``req_to_token``.

Both stages have a Triton implementation (the default, no host syncs) and a
pure-PyTorch reference (``*_ref``); the reference is ground truth in unit tests.
"""

from typing import List, Optional

import torch
import triton
import triton.language as tl


def block_inds_to_kv_inds_for_decoding_ref(
    active_block_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    *,
    block_size: int,
    num_kv_heads: int,
):
    """Pure-PyTorch reference for :func:`block_inds_to_kv_inds_for_decoding`.

    Expand selected block ids into ragged virtual KV indices for decode.

    For each request (outer) and each kv-head (inner) — i.e. batch-major,
    head-minor order matching the ``kv_indptr`` built via
    ``repeat_interleave(num_kv_heads)`` — this expands the selected previous
    blocks plus the current (partial) block into token positions, maps them
    through ``req_to_token`` to physical KV-pool locations, and finally to
    the *virtual* KV coordinate ``virtual_idx = pool_loc * num_kv_heads + h``.

    Padding entries (``-1``) in ``active_block_ids`` are ignored.

    Args:
        active_block_ids: ``[bsz, num_kv_heads, topk]`` int32 logical block
            ids (``-1`` = padding), as returned by the backend's block
            selection.  The current/partial block is NOT included here — it is
            appended by position below.
        seq_lens: ``[bsz]`` total sequence length after the new token.
        req_pool_indices: ``[bsz]`` row index into ``req_to_token``.
        req_to_token: ``[req_pool_size, max_ctx_len]`` int32 token-location map.
        block_size / num_kv_heads: config.

    Returns:
        active_kv_indices: ``[sum_i num_kv_heads * sparse_kv_lens[i]]`` int32,
            ragged, laid out as
            ``req0_h0, req0_h1, ..., req0_h(H-1), req1_h0, ...``.
    """
    H = num_kv_heads
    device = active_block_ids.device

    seq_lens_cpu = seq_lens.tolist()
    req_idx_cpu = req_pool_indices.tolist()
    bsz = len(seq_lens_cpu)
    blk_offsets = torch.arange(block_size, device=device)

    all_inds: List[torch.Tensor] = []
    for i in range(bsz):
        total_len = int(seq_lens_cpu[i])
        req_idx = int(req_idx_cpu[i])
        current_block_id = (total_len - 1) // block_size
        cur_blk_start = current_block_id * block_size

        # Current (partial) block token positions — shared across all heads.
        cur_pos = torch.arange(cur_blk_start, total_len, device=device)

        sel = active_block_ids[i]  # [H, topk]
        for h in range(H):
            ids = sel[h]
            ids = ids[ids >= 0]  # drop padding -> [num_sel]
            if ids.numel() > 0:
                # [num_sel, block_size] token positions of selected prev blocks
                prev_pos = (
                    ids.long().unsqueeze(1) * block_size
                    + blk_offsets.unsqueeze(0)
                ).reshape(-1)
                all_pos = torch.cat([prev_pos, cur_pos])
            else:
                all_pos = cur_pos
            pool_locs = req_to_token[req_idx, all_pos].to(torch.int64)
            virt = pool_locs * H + h
            all_inds.append(virt.to(torch.int32))

    return torch.cat(all_inds)


@triton.jit
def _scatter_kv_inds_kernel(
    # -- selected block ids --
    active_ptr,  # [bsz, H, topk]  int32  (-1 = padding)
    active_stride_b,
    active_stride_h,
    # -- token-location map --
    req_to_token_ptr,  # [req_pool_size, max_ctx]  int32
    req_to_token_stride,
    # -- per-request info --
    req_pool_indices_ptr,  # [bsz]  int64
    seq_lens_ptr,  # [bsz]  int32
    kv_indptr_ptr,  # [bsz*H + 1]  int32 — compact segment offsets
    # -- output --
    kv_indices_ptr,  # [>= kv_indptr[-1]]  int32
    # -- constants --
    block_size: tl.constexpr,
    topk: tl.constexpr,
    H: tl.constexpr,
    TILE: tl.constexpr,  # next_power_of_2(block_size)
):
    """Expand ONE selected block of one virtual request into virtual KV indices.

    Grid ``(bsz, H, topk + 1)``: one program per (request, kv-head, slot).  The
    previous grid was ``(bsz, H)`` with an inner ``for p in range(topk)`` loop —
    at small batch that launched only ``bsz*H`` programs (e.g. 8) and serialized
    the ``topk`` block copies, leaving the GPU mostly idle and latency-bound.
    Promoting the slot loop to the grid's third axis gives ``bsz*H*(topk+1)``
    independent programs (e.g. 264), each doing a single 64-element gather +
    scatter, so the many small memory ops overlap instead of serializing.

    Virtual request ``v = i*H + h`` owns ``kv_indices[kv_indptr[v]:kv_indptr[v+1]]``.
    ``compute_active_block_ids`` guarantees exactly ``actual_topk =
    min(current_block, topk)`` valid blocks per (req, head), ascending, occupying
    slots ``0..actual_topk-1`` with ``-1`` padding strictly at the tail.  We
    exploit that directly: slot ``p`` (``pid_p``):
      * ``p < actual_topk`` -> the p-th selected *previous* block, written at
        offset ``p*block_size``.  No ``b >= 0`` guard is needed — every slot
        below ``actual_topk`` is a valid block by construction, so padding slots
        (``actual_topk <= p < topk``) simply don't load ``b`` and do nothing.
      * ``p == topk`` -> the current (partial) block, written right after the
        selected blocks at offset ``actual_topk*block_size``.
    Per-program scalar recompute (seq_len / req_idx / seg_start) is redundant
    across the topk+1 slots but cheap and fully latency-hidden by the parallelism.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_p = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int32)
    req_idx = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
    current_block = (seq_len - 1) // block_size
    cur_blk_start = current_block * block_size
    actual_topk = tl.minimum(current_block, topk)

    seg_start = tl.load(kv_indptr_ptr + (pid_b * H + pid_h)).to(tl.int64)
    off = tl.arange(0, TILE)

    if pid_p < actual_topk:
        # One selected previous block (slot pid_p) — guaranteed valid (>= 0).
        b = tl.load(
            active_ptr + pid_b * active_stride_b + pid_h * active_stride_h + pid_p
        ).to(tl.int32)
        tok_mask = off < block_size
        pos = b * block_size + off  # token positions within the request
        pool = tl.load(
            req_to_token_ptr + req_idx * req_to_token_stride + pos,
            mask=tok_mask,
            other=0,
        ).to(tl.int64)
        virt = pool * H + pid_h
        tl.store(
            kv_indices_ptr + seg_start + pid_p * block_size + off,
            virt.to(tl.int32),
            mask=tok_mask,
        )
    elif pid_p == topk:
        # Current (partial) block: appended after the selected blocks at offset
        # actual_topk*block_size.
        cur_len = seq_len - cur_blk_start  # tokens in the current (partial) block
        cur_mask = off < cur_len
        cur_pos = cur_blk_start + off
        cur_pool = tl.load(
            req_to_token_ptr + req_idx * req_to_token_stride + cur_pos,
            mask=cur_mask,
            other=0,
        ).to(tl.int64)
        cur_virt = cur_pool * H + pid_h
        tl.store(
            kv_indices_ptr + seg_start + actual_topk * block_size + off,
            cur_virt.to(tl.int32),
            mask=cur_mask,
        )
    # else: padding slot (actual_topk <= pid_p < topk) — nothing to write, and
    # crucially no `b` load.


def block_inds_to_kv_inds_for_decoding(
    active_block_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    *,
    block_size: int,
    num_kv_heads: int,
    topk: int,
    kv_indptr: torch.Tensor = None,
    out: Optional[torch.Tensor] = None,
):
    """Triton implementation of ragged virtual-KV-index expansion.

    Writes each virtual request's segment at the compact ``kv_indptr`` offset.
    ``kv_indptr`` (virtual-batch layout, ``[bsz*H + 1]``) is normally precomputed
    once per step by the backend's decode indices updater and passed in to avoid
    recomputing the cumsum every layer; if ``None`` it is recomputed here.

    ``out`` is the destination buffer.  When provided (the normal backend path),
    indices are written *in place* into it — this is what makes the path
    CUDA-graph capturable: the same fixed buffer is reused every layer / every
    replay, so the flashinfer wrapper's ``_paged_kv_indices_buf`` never has to be
    rebound to a fresh allocation.  When ``None`` (unit tests / ref comparison) a
    fresh buffer is allocated.  Either way the buffer is sized to an upper bound
    so no host sync is needed; flashinfer reads only the valid prefix of each
    segment.

    Returns the destination buffer (``out`` if given), int32, laid out
    batch-major head-minor.
    """
    H = num_kv_heads
    device = active_block_ids.device
    bsz = seq_lens.size(0)

    if kv_indptr is None:
        # Compact per-(req,head) segment offsets (device-only, no sync). This
        # mirrors the decode indices updater so the segment layout matches.
        current_block = (seq_lens.to(torch.int64) - 1) // block_size  # [bsz]
        actual_topk = torch.clamp(current_block, max=topk)  # [bsz]
        cur_block_len = seq_lens.to(torch.int64) - current_block * block_size
        sparse_kv_lens = actual_topk * block_size + cur_block_len  # [bsz]
        virt_seq_lens = sparse_kv_lens.repeat_interleave(H)  # [bsz*H]
        kv_indptr = torch.zeros(bsz * H + 1, dtype=torch.int32, device=device)
        kv_indptr[1:] = torch.cumsum(virt_seq_lens, dim=0).to(torch.int32)
    else:
        # Use the precomputed indptr; only the first bsz*H+1 entries are valid.
        kv_indptr = kv_indptr[: bsz * H + 1]

    # Over-allocate to the upper bound (topk + 1 blocks per virtual request) so
    # buffer sizing needs no host sync.  Only the compact prefix is written/read.
    upper_bound = bsz * H * (topk + 1) * block_size
    if out is None:
        kv_indices = torch.empty(upper_bound, dtype=torch.int32, device=device)
    else:
        # In-place into the caller's fixed buffer (graph-safe).  The buffer may
        # be larger than this step's upper bound (sized for max_bs at capture);
        # only the compact prefix written here is ever read back by flashinfer.
        assert out.numel() >= upper_bound, (
            f"out buffer too small: {out.numel()} < {upper_bound}"
        )
        kv_indices = out

    TILE = triton.next_power_of_2(block_size)
    # Grid (bsz, H, topk + 1): one program per (request, kv-head, slot) — the
    # last slot (pid_p == topk) writes the current/partial block.
    _scatter_kv_inds_kernel[(bsz, H, topk + 1)](
        active_block_ids,
        active_block_ids.stride(0),
        active_block_ids.stride(1),
        req_to_token,
        req_to_token.stride(0),
        req_pool_indices,
        seq_lens,
        kv_indptr,
        kv_indices,
        block_size=block_size,
        topk=topk,
        H=H,
        TILE=TILE,
    )

    # The buffer is over-allocated; flashinfer reads only each segment's valid
    # prefix via the (compact) kv_indptr it was planned with, so the unused tail
    # is never touched and no host sync is needed to trim it.
    return kv_indices


# ===========================================================================
#  Block selection: per-(request, kv-head) top-k by query-summary similarity.
#  Shared by MoBA (group-merged raw query) and SeerAttention-R (RoPE'd gate
#  query).  See module docstring for the representation-agnostic contract.
# ===========================================================================


def _sort_active_ascending(active: torch.Tensor) -> torch.Tensor:
    """Sort selected block ids ascending along the last dim, keeping ``-1``
    padding at the end.

    The downstream scatter writes blocks in active-slot order, so the slots
    must be in increasing block-id (== increasing position) order: this keeps
    each virtual request's ``kv_indices`` segment monotonic in position.
    ``-1`` padding is mapped to a large sentinel before sorting so it lands at
    the tail (preserving the "slot p -> write offset p*block_size" layout the
    scatter relies on).

    Args:
        active: ``[bsz, num_kv_heads, topk]`` int32, ``-1`` = padding.

    Returns:
        Sorted copy, same shape/dtype.
    """
    sentinel = torch.iinfo(torch.int32).max
    keys = torch.where(active < 0, sentinel, active)
    keys, _ = torch.sort(keys, dim=-1)
    return torch.where(keys == sentinel, -1, keys).to(torch.int32)


def compute_active_block_ids_ref(
    q_retrieval: torch.Tensor,
    summary_buffer: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_summary: torch.Tensor,
    *,
    block_size: int,
    topk: int,
    num_init_blocks: int = 1,
    num_local_blocks: int = 0,
):
    """Pure-PyTorch reference for :func:`compute_active_block_ids`.

    Select the active *previous* blocks per (request, kv-head) for decode.

    Per-(kv-head) routing: each kv-head scores its (caller-prepared) query
    against the cached block summaries and keeps the top-k blocks.  How the
    query is prepared from the model query is the caller's representation choice
    and is intentionally *not* part of this function (MoBA mean-merges the GQA
    group; SeerAttention-R projects + RoPEs the gate query).  Two contiguous
    bands of blocks are *always* kept (shared across kv-heads):

    * the first ``num_init_blocks`` blocks ``[0, num_init_blocks)`` — the
      attention-sink band;
    * the last ``num_local_blocks`` *previous* blocks
      ``[num_prev - num_local_blocks, num_prev)`` — the local band, i.e. the
      blocks immediately before the current block (the current/self block is
      excluded; it is always attended and appended separately downstream).

    The remaining ``topk - nf`` slots are filled by query-summary similarity
    from the non-forced candidates.  The *current* (last, partial) block is NOT
    included here — it is appended later by
    :func:`block_inds_to_kv_inds_for_decoding` by position.

    The number of selected blocks per (req, head) equals
    ``actual_topk = min(current_block_id, topk)`` so the produced kv length
    matches ``BlockSparseIndicesUpdaterDecode._compute_sparse_kv_lens``.

    Args:
        q_retrieval: ``[bsz, num_kv_heads, head_dim]`` — the per-(kv-head)
            query, already prepared by the caller.
        summary_buffer: ``[summary_pool_size+1, num_kv_heads, head_dim]``
            (slot 0 reserved as padding).
        seq_lens: ``[bsz]`` total sequence length after the new token.
        req_pool_indices: ``[bsz]`` row index into ``req_to_summary``.
        req_to_summary: ``[req_pool_size, max_num_summary]`` int32 mapping
            ``(req, logical_block_id) -> summary slot``.
        block_size / topk: config.
        num_init_blocks: number of leading sink blocks to always keep (>= 0).
            Defaults to ``1`` (the classic attention-sink block).
        num_local_blocks: number of trailing *previous* blocks (excluding the
            current/self block) to always keep (>= 0).  Defaults to ``0``.
            If the init and local bands overlap (short sequences), the union is
            de-duplicated.

    Returns:
        active_block_ids: ``[bsz, num_kv_heads, topk]`` int32, holding the
            selected *logical* block ids.  Unused slots are padded with
            ``-1`` (only happens when ``current_block_id < topk``).
    """
    assert num_init_blocks >= 0 and num_local_blocks >= 0, (
        "num_init_blocks and num_local_blocks must be non-negative, got "
        f"{num_init_blocks}, {num_local_blocks}"
    )

    bsz, H, d = q_retrieval.shape
    device = q_retrieval.device

    active = torch.full((bsz, H, topk), -1, dtype=torch.int32, device=device)

    q_retrieval = q_retrieval.float()

    seq_lens_cpu = seq_lens.tolist()
    req_idx_cpu = req_pool_indices.tolist()

    for i in range(bsz):
        total_len = int(seq_lens_cpu[i])
        req_idx = int(req_idx_cpu[i])
        current_block_id = (total_len - 1) // block_size
        num_prev = current_block_id  # number of complete previous blocks

        if num_prev <= 0:
            # Whole sequence lives in block 0 (== current block); nothing to
            # select. The current block is handled by position downstream.
            continue

        if num_prev <= topk:
            # Dense over previous blocks: keep all of blocks [0, num_prev).
            # Both forced bands are subsets of these, so nothing special to do.
            ids = torch.arange(num_prev, dtype=torch.int32, device=device)
            active[i, :, :num_prev] = ids.unsqueeze(0).expand(H, num_prev)
            continue

        # --- Sparse selection ------------------------------------------------
        # Build the forced set: init band [0, num_init) then local band
        # [num_prev - num_local, num_prev), clamped to [0, num_prev) and
        # de-duplicated (init first, preserving order). Shared across heads.
        forced: List[int] = []
        seen = set()
        n_init = min(num_init_blocks, num_prev)
        for b in range(n_init):
            if b not in seen:
                seen.add(b)
                forced.append(b)
        local_start = max(0, num_prev - num_local_blocks)
        for b in range(local_start, num_prev):
            if b not in seen:
                seen.add(b)
                forced.append(b)
        nf = len(forced)

        if nf >= topk:
            # Forced bands alone fill (or overflow) the budget; keep the
            # first ``topk`` of them. Preserves the topk count invariant.
            chosen = torch.tensor(
                forced[:topk], dtype=torch.int32, device=device
            )
            active[i, :, :topk] = chosen.unsqueeze(0).expand(H, topk)
            continue

        if nf > 0:
            forced_t = torch.tensor(forced, dtype=torch.int32, device=device)
            active[i, :, :nf] = forced_t.unsqueeze(0).expand(H, nf)

        # Candidate blocks = [0, num_prev) minus the forced ones.  Score them
        # per-head and take the top ``topk - nf``.
        n_pick = topk - nf
        cand_mask = torch.ones(num_prev, dtype=torch.bool, device=device)
        if nf > 0:
            cand_mask[forced_t.long()] = False
        cand_block_ids = torch.arange(num_prev, device=device)[cand_mask]  # [num_cand]
        cand_slots = req_to_summary[req_idx, cand_block_ids].long()  # [num_cand]
        cand_summ = summary_buffer[cand_slots].float()  # [num_cand, H, d]
        # scores[h, p] = <q_retrieval[i, h], summary_of_candidate_p[h]>
        scores = torch.einsum("hd,phd->hp", q_retrieval[i], cand_summ)  # [H, num_cand]
        sel = scores.topk(n_pick, dim=1).indices  # [H, n_pick] (candidate space)
        picked = cand_block_ids[sel].to(torch.int32)  # [H, n_pick] logical block ids
        active[i, :, nf:topk] = picked

    # Sort selected block ids ascending per (req, head) so each virtual
    # request's kv_indices segment is monotonic in position. -1 padding is kept
    # at the end by sorting it
    # as +inf (so the "slot p -> offset p*block_size" scatter layout holds).
    active = _sort_active_ascending(active)
    return active


@triton.jit
def _score_blocks_kernel(
    # -- per-(kv-head) query (caller-prepared) --
    q_ptr,  # [bsz, H, D]  (model dtype)
    q_stride_b,
    q_stride_h,
    # -- summary buffer --
    summary_buf_ptr,  # [summary_pool_size+1, H, D]
    summary_buf_stride_slot,
    summary_buf_stride_head,
    # -- block -> summary slot map --
    req_to_summary_ptr,  # [req_pool_size, max_num_summary]  int32
    req_to_summary_stride,
    # -- per-request info --
    req_pool_indices_ptr,  # [bsz]  int64
    seq_lens_ptr,  # [bsz]  int32
    # -- output scores --
    scores_ptr,  # [bsz, H, NBLK]  float32  (-inf for non-candidate)
    scores_stride_b,
    scores_stride_h,
    # -- constants --
    block_size: tl.constexpr,
    num_init_blocks: tl.constexpr,
    num_local_blocks: tl.constexpr,
    D: tl.constexpr,
    D_BLOCK: tl.constexpr,
    NBLK: tl.constexpr,  # next_power_of_2(max_num_summary) = scores width
    BN: tl.constexpr,  # block-tile size (fixed -> constant register footprint)
):
    """Score a tile of ``BN`` candidate blocks for one (request, kv-head).

    Grid: ``(bsz, H, num_block_tiles)``.  Each program handles ``BN`` blocks and
    does ONE 2D load ``summary[slots_tile, h, :]`` of shape ``[BN, D_BLOCK]``;
    the D dimension is contiguous in the summary buffer so the inner reads are
    coalesced, and the dot product over D is a vectorized reduction.  The
    register footprint is ``[BN, D_BLOCK]`` regardless of context length, so it
    never spills as the number of blocks grows.

    Non-candidate blocks (outside ``[0, num_prev)`` or inside the forced
    init/local bands) are written as ``-inf`` so the downstream selection never
    picks them by score.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int32)
    req_idx = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
    num_prev = (seq_len - 1) // block_size # complete previous blocks, which is the current block id

    # Early-return for over-launched tiles: this tile's lowest block id is
    # already >= num_prev, so every lane is a non-candidate.  The caller
    # pre-fills `scores` with -inf, so skipping the stores leaves exactly the
    # -inf the selection expects — no work, no writes.  This bounds the score
    # kernel's effective grid to each request's real block count instead of the
    # batch-wide NBLK padding (short requests in a ragged batch do far less).
    if pid_t * BN >= num_prev:
        return

    local_start = num_prev - num_local_blocks

    row = pid_t * BN + tl.arange(0, BN)  # [BN] block ids in this tile
    is_cand = (row >= num_init_blocks) & (row < local_start) & (row < num_prev)

    # summary slot for each block in the tile
    slot = tl.load(
        req_to_summary_ptr + req_idx * req_to_summary_stride + row,
        mask=is_cand,
        other=0,
    ).to(tl.int64)  # [BN]

    # query vector [D_BLOCK] (loaded once per program)
    d_off = tl.arange(0, D_BLOCK)
    d_in = d_off < D
    q_vec = tl.load(
        q_ptr + pid_b * q_stride_b + pid_h * q_stride_h + d_off,
        mask=d_in,
        other=0.0,
    ).to(tl.float32)  # [D_BLOCK]

    # [BN, D_BLOCK] coalesced over D, then vectorized dot product over D.
    s_ptr = (
        summary_buf_ptr
        + slot[:, None] * summary_buf_stride_slot
        + pid_h * summary_buf_stride_head
        + d_off[None, :]
    )
    s_tile = tl.load(
        s_ptr,
        mask=is_cand[:, None] & d_in[None, :],
        other=0.0,
    ).to(tl.float32)  # [BN, D_BLOCK]
    score = tl.sum(s_tile * q_vec[None, :], axis=1)  # [BN]
    score = tl.where(is_cand, score, float("-inf"))

    tl.store(
        scores_ptr + pid_b * scores_stride_b + pid_h * scores_stride_h + row,
        score,
        mask=row < NBLK,
    )


@triton.jit
def _select_from_scores_kernel(
    # -- candidate scores --
    scores_ptr,  # [bsz, H, NBLK]  float32  (-inf for non-candidate)
    scores_stride_b,
    scores_stride_h,
    # -- per-request info --
    seq_lens_ptr,  # [bsz]  int32
    # -- output --
    active_ptr,  # [bsz, H, topk]  int32  (-1 = padding), ascending
    active_stride_b,
    active_stride_h,
    # -- constants --
    block_size: tl.constexpr,
    topk: tl.constexpr,
    n_pick: tl.constexpr,
    num_init_blocks: tl.constexpr,
    num_local_blocks: tl.constexpr,
    NBLK: tl.constexpr,  # next_power_of_2(max_num_summary)
):
    """Select active blocks for one (request, kv-head) from precomputed scores.

    Steps (all over the [NBLK] block-id axis):
      1. select = forced init/local bands ∪ top-``n_pick`` candidates by score.
         A score threshold (the n_pick-th largest) is found via ``tl.sort`` (only
         the scalar value is used, not its index).  Strict winners (``score >
         thr``) are all kept; threshold ties (``score == thr``) fill the
         remaining quota by ascending id, so the result matches ``torch.topk``'s
         "keep the highest scores" even when several blocks sit exactly at the
         threshold.  Ties *among* the equal-to-threshold blocks are broken by
         ascending id (arbitrary but stable) — fine since tied blocks are equally
         relevant.  The dense regime (``num_prev <= topk``: keep all previous
         blocks) needs no separate branch — with <= n_pick candidates the
         threshold drops to (or below) the smallest candidate score, so every
         candidate is picked and the union covers all of ``[0, num_prev)``.
      2. compact the mask to ascending block ids via cumulative sum and store.
         Because block ids are scanned in increasing order, the output is
         naturally ascending (required by the scatter's slot->offset layout),
         with no sort over ids.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int32)
    current_block = (seq_len - 1) // block_size
    num_prev = current_block
    local_start = num_prev - num_local_blocks

    blk = tl.arange(0, NBLK)

    # prefill -1 to all active_ptr
    tl.store(
        active_ptr + pid_b * active_stride_b + pid_h * active_stride_h + blk,
        tl.full([NBLK], -1, dtype=tl.int32),
        mask=blk < topk, # topk is not necessarily a power of 2, but blk is
    )

    # Early-return when the request has no previous blocks (block 0)
    if num_prev <= 0:
        return

    acc = tl.load(
        scores_ptr + pid_b * scores_stride_b + pid_h * scores_stride_h + blk,
    )  # [NBLK]; non-candidate lanes (forced bands, >= num_prev, padding) are -inf

    # find the threshold score
    if n_pick > 0:
        sorted_desc = tl.sort(acc, descending=True)  # [NBLK]
        thr = tl.gather(
            sorted_desc, tl.full([NBLK], n_pick - 1, tl.int32), axis=0
        )  # [NBLK], every lane == sorted_desc[n_pick-1]
    else:
        thr = tl.full([NBLK], float("inf"), tl.float32)  # no score-picked blocks

    # Pick exactly n_pick candidates with the highest scores. Ties are broken by ascending id.
    neg_inf = float("-inf")
    gt = acc > thr
    eq = (acc == thr) & (thr > neg_inf)
    n_gt = tl.sum(gt.to(tl.int32))
    need = n_pick - n_gt  # remaining slots to fill from the tied blocks
    eq_rank = tl.cumsum(eq.to(tl.int32)) - eq.to(tl.int32)  # exclusive prefix
    picked = gt | (eq & (eq_rank < need))

    # select blocks from init band, local band and n_pick candidates
    sel = (
        ((blk < num_init_blocks) & (blk < num_prev))  # init band
        | ((blk >= local_start) & (blk < num_prev))  # local band
        | picked # n_pick candidates between init and local bands
    )

    # compact selected block ids ascending (cumsum gives write slot)
    sel_i = sel.to(tl.int32)
    pos = tl.cumsum(sel_i) - sel_i
    tl.store(
        active_ptr + pid_b * active_stride_b + pid_h * active_stride_h + pos,
        blk.to(tl.int32),
        # pos < topk is an invariant here (|sel| <= topk in both regimes); kept
        # as a cheap out-of-bounds guard so a future forced-band misconfig
        # degrades to a dropped entry instead of corrupting the next row.
        mask=sel & (pos < topk),
    )

@triton.jit
def _select_from_scores_radix_kernel(
    # -- candidate scores --
    scores_ptr,  # [bsz, H, NBLK]  float32  (-inf for non-candidate)
    scores_stride_b,
    scores_stride_h,
    # -- per-request info --
    seq_lens_ptr,  # [bsz]  int32
    # -- output --
    active_ptr,  # [bsz, H, topk]  int32  (-1 = padding), ascending
    active_stride_b,
    active_stride_h,
    # -- constants --
    block_size: tl.constexpr,
    topk: tl.constexpr,
    n_pick: tl.constexpr,
    num_init_blocks: tl.constexpr,
    num_local_blocks: tl.constexpr,
    NBLK: tl.constexpr,  # next_power_of_2(max_num_summary)
):
    """Sort-free variant of :func:`_select_from_scores_kernel` (binary B-plan).

    Identical contract and output to ``_select_from_scores_kernel`` — only the
    *threshold* (the ``n_pick``-th largest candidate score) is found differently:

      * ``_select_from_scores_kernel`` does ``tl.sort(acc)`` (bitonic, cost
        ``O(NBLK·log²NBLK)``) and reads off element ``n_pick-1``.
      * this kernel does a **bit-serial radix-select**: map each fp32 score to an
        order-preserving uint32 key, then walk the 32 key bits MSB→LSB, at each
        bit counting how many candidate keys fall in the "this bit = 1" half of
        the still-undecided range with one ``tl.sum``.  Cost ``O(32·NBLK)`` — a
        fixed 32 reduction passes, no ``log²`` factor — so it scales far better as
        NBLK grows (NBLK=16384: log²≈196 vs 32).  The reconstructed key is the
        exact key of the ``n_pick``-th largest element (the map is a bijection),
        so converting it back to float gives the exact same threshold value the
        sort path would, and the downstream gt/eq quota fill is byte-identical.

    fp32→uint32 order-preserving map (IEEE-754): positive floats (sign bit 0) get
    their sign bit set; negatives (sign bit 1) get every bit flipped.  As uint32
    this orders exactly like the floats (``-inf`` -> smallest key), so "k-th
    largest float" == "k-th largest key".  The map is bijective, so the selected
    key reconstructs to the score it came from with no rounding.

    Everything after the threshold (forced bands ∪ gt/eq quota, ascending cumsum
    compaction, in-kernel -1 clear, early-return) is copied verbatim from
    ``_select_from_scores_kernel`` — see that kernel for the detailed rationale.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int32)
    current_block = (seq_len - 1) // block_size
    num_prev = current_block
    local_start = num_prev - num_local_blocks

    blk = tl.arange(0, NBLK)

    # Clear this (req, head) output row to -1 padding (in-kernel, graph-safe).
    tl.store(
        active_ptr + pid_b * active_stride_b + pid_h * active_stride_h + blk,
        tl.full([NBLK], -1, dtype=tl.int32),
        mask=blk < topk,
    )
    if num_prev <= 0:
        return

    acc = tl.load(
        scores_ptr + pid_b * scores_stride_b + pid_h * scores_stride_h + blk,
    )  # [NBLK]; non-candidate lanes (forced bands, >= num_prev, padding) are -inf

    neg_inf = float("-inf")

    if n_pick > 0:
        # --- bit-serial radix-select for the n_pick-th largest score ----------
        # Order-preserving fp32 -> uint32 key.  -inf lanes (non-candidates) map
        # to the smallest key under this transform, so they are naturally below
        # every finite candidate — no separate masking of `key` is needed, and a
        # threshold that falls onto a -inf lane reconstructs back to exactly -inf
        # (the dense regime: fewer candidates than n_pick -> thr == -inf -> every
        # candidate clears `gt`, matching the sort path).  The select runs over
        # ALL NBLK lanes (padding included as -inf), exactly like the sort path
        # which sorts the full row and reads element n_pick-1.
        ix = acc.to(tl.uint32, bitcast=True)
        sign_mask = tl.where(
            (ix >> 31) != 0,
            tl.full([NBLK], 0xFFFFFFFF, tl.uint32),
            tl.full([NBLK], 0x80000000, tl.uint32),
        )
        key = ix ^ sign_mask  # uint32 order == float order; -inf -> 0x007FFFFF

        # Find the n_pick-th largest key (1-indexed k) bit by bit, MSB->LSB.
        prefix = tl.zeros([], tl.uint32)  # decided high bits of the answer
        k = n_pick
        for b in tl.static_range(31, -1, -1):
            bit = tl.full([], 1, tl.uint32) << b
            # Count lanes whose bits strictly above b equal `prefix` (the already-
            # decided high bits) AND whose bit b is 1 — those land in the upper
            # (bit=1) half of the still-undecided range.
            above = key & (~((bit << 1) - 1))  # zero out bits b..0, keep bits >b
            in_branch = (above == prefix) & ((key & bit) != 0)
            cnt = tl.sum(in_branch.to(tl.int32))
            take_one = cnt >= k  # the k-th largest lies in this (bit=1) upper half
            prefix = tl.where(take_one, prefix | bit, prefix)
            k = tl.where(take_one, k, k - cnt)
        thr_key = prefix
        # uint32 key -> fp32 threshold (invert the order-preserving map).
        inv_mask = tl.where(
            (thr_key >> 31) != 0,
            tl.full([], 0x80000000, tl.uint32),
            tl.full([], 0xFFFFFFFF, tl.uint32),
        )
        thr_ix = thr_key ^ inv_mask
        thr_s = thr_ix.to(tl.float32, bitcast=True)
        thr = thr_s + 0.0 * acc  # broadcast scalar -> [NBLK]
    else:
        thr = tl.full([NBLK], float("inf"), tl.float32)  # no score-picked blocks

    # ---- selection tail: identical to _select_from_scores_kernel ----
    gt = acc > thr
    eq = (acc == thr) & (thr > neg_inf)
    n_gt = tl.sum(gt.to(tl.int32))
    need = n_pick - n_gt
    eq_rank = tl.cumsum(eq.to(tl.int32)) - eq.to(tl.int32)
    picked = gt | (eq & (eq_rank < need))

    sel = (
        ((blk < num_init_blocks) & (blk < num_prev))  # init band
        | ((blk >= local_start) & (blk < num_prev))  # local band
        | picked
    )
    sel_i = sel.to(tl.int32)
    pos = tl.cumsum(sel_i) - sel_i
    tl.store(
        active_ptr + pid_b * active_stride_b + pid_h * active_stride_h + pos,
        blk.to(tl.int32),
        mask=sel & (pos < topk),
    )

def compute_active_block_ids(
    q_retrieval: torch.Tensor,  # [bsz, num_kv_heads, head_dim]  (caller-prepared)
    summary_buffer: torch.Tensor,  # [summary_pool_size+1, num_kv_heads, head_dim]
    seq_lens: torch.Tensor,  # [bsz]
    req_pool_indices: torch.Tensor,  # [bsz]
    req_to_summary: torch.Tensor,  # [req_pool_size, max_num_summary]
    *,
    block_size: int,
    topk: int,
    num_init_blocks: int = 1,
    num_local_blocks: int = 0,
):
    """Triton block selection in two stages (see ``*_ref`` for exact semantics).

    Stage 1 (``_score_blocks_kernel``): score every candidate block with a
    coalesced 2D summary load, written to a ``[bsz, H, NBLK]`` scratch.  The
    per-program register footprint is fixed at ``[BN, D_BLOCK]`` so it never
    spills as context length (NBLK) grows; over-launched tiles past a request's
    real block count early-return (no wasted loads/stores in a ragged batch).

    Stage 2 (``_select_from_scores_radix_kernel``): per (req, head), pick the
    forced bands + top-``n_pick`` candidates by score and emit ascending block
    ids.  The threshold is found by radix-select (sort-free); the equivalent
    ``_select_from_scores_kernel`` (tl.sort) is kept as the parity reference.

    ``q_retrieval`` is the per-(kv-head) query already prepared by the caller
    (GQA-group merge for MoBA; gate projection + RoPE for SeerAttention-R — both
    representation choices are kept out of these kernels).  Ties at the score
    threshold are broken arbitrarily.  No host syncs.
    """
    assert num_init_blocks >= 0 and num_local_blocks >= 0, (
        "num_init_blocks and num_local_blocks must be non-negative, got "
        f"{num_init_blocks}, {num_local_blocks}"
    )
    assert num_init_blocks + num_local_blocks <= topk, (
        "num_init_blocks + num_local_blocks must be <= topk, got "
        f"{num_init_blocks} + {num_local_blocks} > {topk}"
    )

    bsz, num_kv_heads, head_dim = q_retrieval.shape
    device = q_retrieval.device
    num_force = num_init_blocks + num_local_blocks
    n_pick = topk - num_force

    q_retrieval = q_retrieval.contiguous()
    nblk = req_to_summary.size(1)  # max_num_summary
    NBLK = triton.next_power_of_2(nblk)
    # NOTE: NBLK is a STATIC upper bound — next_power_of_2(ceil(model_max_ctx /
    # block_size)) (e.g. 40960/64 -> 640 -> NBLK=1024), independent of the current
    # batch's real lengths.  Stage 2 runs its sort/cumsum/reductions over this
    # full width regardless of how short the batch actually is.  See the
    # consolidated optimization TODO at the Stage-2 launch below.
    D_BLOCK = triton.next_power_of_2(head_dim)
    BN = 64  # block-tile size; [BN, D_BLOCK] fits comfortably in registers/SRAM

    # Stage 1: scores scratch (-inf default covers padding lanes).
    scores = torch.full(
        (bsz, num_kv_heads, NBLK), float("-inf"), dtype=torch.float32, device=device
    )
    n_tiles = triton.cdiv(NBLK, BN)
    _score_blocks_kernel[(bsz, num_kv_heads, n_tiles)](
        q_retrieval,
        q_retrieval.stride(0),
        q_retrieval.stride(1),
        summary_buffer,
        summary_buffer.stride(0),
        summary_buffer.stride(1),
        req_to_summary,
        req_to_summary.stride(0),
        req_pool_indices,
        seq_lens,
        scores,
        scores.stride(0),
        scores.stride(1),
        block_size=block_size,
        num_init_blocks=num_init_blocks,
        num_local_blocks=num_local_blocks,
        D=head_dim,
        D_BLOCK=D_BLOCK,
        NBLK=NBLK,
        BN=BN,
    )

    # Stage 2: select per (req, head).
    #
    # ====================================================================
    # CONSOLIDATED OPTIMIZATION TODO (perf, long-context).  Two independent,
    # stackable axes for this stage, plus a note on when CUDA C is actually
    # warranted.  At the CURRENT 40K context cap (NBLK=1024) this stage is near
    # the launch-bound floor (~11.5us/call, microbench) and ~4.5% of the decode
    # step, so NONE of this moves end-to-end yet — the value is at long context
    # (NBLK >= 2048, i.e. >= ~128K tokens).  Recorded so the analysis isn't lost.
    #
    # AXIS 1 — sort-free threshold (DONE, now the DEFAULT path):
    #   The threshold (n_pick-th-largest score) is found by
    #   `_select_from_scores_radix_kernel` — a 32-pass bit-serial radix-select,
    #   O(32·NBLK), no log² factor.  The older `_select_from_scores_kernel` uses
    #   `tl.sort` (O(NBLK·log²NBLK) bitonic) and is KEPT as the parity reference
    #   (seer_attn_radix_parity.py: 110/110 byte-identical) and a fallback; it is
    #   no longer on the live path.  Microbench (seer_attn_select_sort_vs_radix.py)
    #   showed radix never worse at the current NBLK=1024 (equal @bsz<=8 — both at
    #   the launch-bound floor — and 1.59x @bsz=32), and 2.1x@2048 .. 3.4x@16384
    #   as NBLK grows.  Pure Triton — no CUDA C needed (only a byte-level
    #   histogram-radix would benefit from CUDA C).
    #
    # AXIS 2 — shrink NBLK to the batch's real max (NOT done):
    #   NBLK is a STATIC upper bound = next_pow2(ceil(model_max_ctx/block_size))
    #   (40960/64 -> 1024), independent of batch lengths.  An 8K-max batch (real
    #   summary length 128) still pays the 1024-wide sort/cumsum/reductions — up
    #   to ~8x waste + sort's log² on 1024 not 128.  Stage 1 already avoids this
    #   via grid-tile early-return (`pid_t*BN >= num_prev`); stage 2 does not.
    #   Triton fix: pass NBLK_eff = next_pow2(ceil(seq_lens_cpu.max()/block_size))
    #   as the constexpr (seq_lens_cpu is already on forward_batch — NO extra
    #   device sync).  CAVEAT: NBLK is a `constexpr`, so a per-batch NBLK means a
    #   per-batch re-launch; under CUDA graph NBLK is frozen at capture, so the
    #   Triton version can only shrink in EAGER mode (or needs per-captured-bs
    #   NBLK buckets) — graph replay can't change it.
    #
    # WHEN CUDA C IS ACTUALLY WARRANTED (the one thing Triton can't do):
    #   The constexpr-NBLK constraint is what forces the "static NBLK (graph-safe
    #   but wasteful) XOR per-launch NBLK (efficient but breaks graph)" dilemma.
    #   CUDA C dissolves it: fixed launch dims (grid=(bsz,H), fixed block) with a
    #   RUNTIME loop bound `for (i = tid; i < num_prev; i += blockDim.x)` reading
    #   num_prev from `seq_lens` in-kernel.  This gets per-request work shrink
    #   (Axis 2's win) AND graph compatibility AT ONCE — graph capture records the
    #   launch, not the data, so replay re-reads seq_lens and does the new amount
    #   of work (exactly how flashinfer's paged attention is graph-compatible).
    #   That dynamic-bound + graph combination is the ONLY thing here Triton
    #   fundamentally cannot express — it is the real reason to reach for CUDA C.
    #   NOT a reason: a plain Triton->CUDA C transliteration (no algorithm/bound
    #   change) buys ~0% — Triton compiles to PTX/SASS within a few % of hand C
    #   for this load/sort/scan/scatter pattern, and at current sizes the kernel
    #   is launch-bound (transliteration doesn't cut launch overhead).  The radix
    #   3x is ALGORITHMIC (already captured in Triton above); CUDA C on top buys
    #   only a further ~1.3-1.5x from shared-mem/warp-ballot radix, and only at
    #   large NBLK.  At the current 40K cap, even a perfect CUDA C kernel is
    #   end-to-end imperceptible (still hits the ~11.5us launch floor).
    #
    # SMALLEST CURRENT-REGIME LEVER (orthogonal to both axes): the stage is
    # launch-bound, so fusing score+select+scatter to cut the launch COUNT is the
    # only thing with a chance of being measurable today — and that's partly
    # doable in Triton without CUDA C.
    # ====================================================================
    #
    # No host-side init needed: the kernel clears each row to -1 before
    # scattering (self-contained under CUDA graph — see the kernel).
    active = torch.empty((bsz, num_kv_heads, topk), dtype=torch.int32, device=device)
    _select_from_scores_radix_kernel[(bsz, num_kv_heads)](
        scores,
        scores.stride(0),
        scores.stride(1),
        seq_lens,
        active,
        active.stride(0),
        active.stride(1),
        block_size=block_size,
        topk=topk,
        n_pick=n_pick,
        num_init_blocks=num_init_blocks,
        num_local_blocks=num_local_blocks,
        NBLK=NBLK,
    )
    return active
