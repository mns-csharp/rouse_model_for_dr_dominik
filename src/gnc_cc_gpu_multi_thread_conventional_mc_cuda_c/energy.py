"""Energy computation — torch CPU tensors.

3-zone potential, all-pairs delta-E and dense BxB rank-1 EMM correction
done with `torch.cdist` + bool masking. Fast for moderate N*K because
oneDNN's GEMM is well-tuned even single-threaded.

Sequence-separation skips: Calpha-Calpha < min_seq_caca, SG-SG < min_seq_sgsg,
Calpha-SG < min_seq_casg.
"""

from __future__ import annotations

import numpy as np
import torch


def _zone_e(r2: torch.Tensor, r_rep_sq: float, r_max_sq: float,
            rep_e: float, contact_e: float) -> torch.Tensor:
    e = torch.where(r2 < r_rep_sq, torch.full_like(r2, rep_e), torch.zeros_like(r2))
    if contact_e != 0.0:
        e = torch.where((r2 >= r_rep_sq) & (r2 < r_max_sq),
                        torch.full_like(r2, contact_e), e)
    return e


def _all_beads_flat(ca, sg):
    """Interleave [Ca0, SG0, Ca1, SG1, ...] -> [n_chains, 2N, 3]."""
    n_chains, N, _ = ca.shape
    flat = torch.empty(n_chains, 2 * N, 3, dtype=ca.dtype, device=ca.device)
    flat[:, 0::2, :] = ca
    flat[:, 1::2, :] = sg
    return flat


def batch_delta_e_torch(
        old_ca, new_ca, old_sg, new_sg, n_moved,
        chain_idx, bead_start,
        ca_full, sg_full,
        N, box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg,
):
    """Compute per-proposal delta-E by direct all-pairs evaluation.

    For single-core simplicity we evaluate against ALL non-moved beads in
    the same chain + ALL beads in other chains (no cell list — torch.cdist
    is fast enough for the smoke-test sizes).
    """
    B = old_ca.shape[0]
    device = old_ca.device
    dtype = old_ca.dtype
    inv_box = 1.0 / box
    n_chains = ca_full.shape[0]

    # Flatten Cα and SG separately to avoid mixed-type mask logic.
    ca_flat = ca_full.reshape(-1, 3)            # [n_chains*N, 3]
    sg_flat = sg_full.reshape(-1, 3)
    delta_e = torch.zeros(B, dtype=dtype, device=device)

    for b in range(B):
        nm = int(n_moved[b].item())
        if nm == 0:
            continue
        ci = int(chain_idx[b].item())
        ms = int(bead_start[b].item())

        old_ca_b = old_ca[b, :nm]    # [nm, 3]
        new_ca_b = new_ca[b, :nm]
        old_sg_b = old_sg[b, :nm]
        new_sg_b = new_sg[b, :nm]
        moved_residues = torch.arange(ms, ms + nm, device=device)

        # All other Calpha beads (exclude moved residues from this chain).
        all_ca_idx = torch.arange(n_chains * N, device=device)
        ca_chain = all_ca_idx // N
        ca_resid = all_ca_idx % N
        # Build "skip" masks per pair-type using sequence separation.
        # Calpha vs moved Calpha (range mod):
        seq_diff = (ca_resid.unsqueeze(0) - moved_residues.unsqueeze(1)).abs()  # [nm, M]
        skip_caca = (ca_chain.unsqueeze(0) == ci) & (seq_diff < min_caca)        # [nm, M]
        # Mark moved Calpha residues themselves (always skip — they're being moved).
        skip_self = (ca_chain.unsqueeze(0) == ci) & (
            (ca_resid.unsqueeze(0) >= ms) & (ca_resid.unsqueeze(0) < ms + nm))
        keep_caca = ~(skip_caca | skip_self)

        # delta_e (Cα-Cα, moved Cα vs other Cα).
        cv = ca_flat.unsqueeze(0)
        d_old = cv - old_ca_b.unsqueeze(1)
        d_old -= box * torch.round(d_old * inv_box)
        r2_old = (d_old * d_old).sum(dim=-1)
        d_new = cv - new_ca_b.unsqueeze(1)
        d_new -= box * torch.round(d_new * inv_box)
        r2_new = (d_new * d_new).sum(dim=-1)
        e_old = (_zone_e(r2_old, r_rep_sq, r_max_sq, rep_e, contact_e) * keep_caca).sum()
        e_new = (_zone_e(r2_new, r_rep_sq, r_max_sq, rep_e, contact_e) * keep_caca).sum()

        # Calpha-SG (moved Cα vs all SG, excluding intra-residue and moved SG).
        sg_chain = ca_chain
        sg_resid = ca_resid
        skip_casg = (sg_chain.unsqueeze(0) == ci) & (seq_diff < min_casg)
        skip_self_sg = (sg_chain.unsqueeze(0) == ci) & (
            (sg_resid.unsqueeze(0) >= ms) & (sg_resid.unsqueeze(0) < ms + nm))
        keep_casg_a = ~(skip_casg | skip_self_sg)
        sv = sg_flat.unsqueeze(0)
        d_old_a = sv - old_ca_b.unsqueeze(1)
        d_old_a -= box * torch.round(d_old_a * inv_box)
        r2_old_a = (d_old_a * d_old_a).sum(dim=-1)
        d_new_a = sv - new_ca_b.unsqueeze(1)
        d_new_a -= box * torch.round(d_new_a * inv_box)
        r2_new_a = (d_new_a * d_new_a).sum(dim=-1)
        e_old += (_zone_e(r2_old_a, r_rep_sq, r_max_sq, rep_e, contact_e) * keep_casg_a).sum()
        e_new += (_zone_e(r2_new_a, r_rep_sq, r_max_sq, rep_e, contact_e) * keep_casg_a).sum()

        # Moved SG vs all Cα (same sequence rules).
        d_old_b = cv - old_sg_b.unsqueeze(1)
        d_old_b -= box * torch.round(d_old_b * inv_box)
        r2_old_b = (d_old_b * d_old_b).sum(dim=-1)
        d_new_b = cv - new_sg_b.unsqueeze(1)
        d_new_b -= box * torch.round(d_new_b * inv_box)
        r2_new_b = (d_new_b * d_new_b).sum(dim=-1)
        e_old += (_zone_e(r2_old_b, r_rep_sq, r_max_sq, rep_e, contact_e) * keep_casg_a).sum()
        e_new += (_zone_e(r2_new_b, r_rep_sq, r_max_sq, rep_e, contact_e) * keep_casg_a).sum()

        # Moved SG vs all SG (excluding intra-residue and moved SG).
        skip_sgsg = (sg_chain.unsqueeze(0) == ci) & (seq_diff < min_sgsg)
        keep_sgsg = ~(skip_sgsg | skip_self_sg)
        d_old_c = sv - old_sg_b.unsqueeze(1)
        d_old_c -= box * torch.round(d_old_c * inv_box)
        r2_old_c = (d_old_c * d_old_c).sum(dim=-1)
        d_new_c = sv - new_sg_b.unsqueeze(1)
        d_new_c -= box * torch.round(d_new_c * inv_box)
        r2_new_c = (d_new_c * d_new_c).sum(dim=-1)
        e_old += (_zone_e(r2_old_c, r_rep_sq, r_max_sq, rep_e, contact_e) * keep_sgsg).sum()
        e_new += (_zone_e(r2_new_c, r_rep_sq, r_max_sq, rep_e, contact_e) * keep_sgsg).sum()

        delta_e[b] = e_new - e_old
    return delta_e


def delta_e_segment_b1(scratch, ca_t, sg_t, gs: int,
                       N: int, n_chains: int, box: float,
                       r_rep_sq, r_max_sq, rep_e, contact_e,
                       min_caca, min_casg, min_sgsg):
    """Single fused CUDA delta-E kernel for one segment (B=1).

    All persistent tensors come from `scratch` with cached cupy views;
    per-segment metadata (chain_idx, bead_start, n_moved_out) is sliced
    in cupy land off pre-cached column views, so there are zero dlpack
    conversions per call.
    """
    import cupy as cp
    from . import cuda_kernels as ck

    cp_de = scratch.cp("delta_e")
    cp_ca = scratch.cp_state_ca(ca_t)
    cp_sg = scratch.cp_state_sg(sg_t)
    cp_old_ca = scratch.cp("old_ca")
    cp_new_ca = scratch.cp("new_ca")
    cp_old_sg = scratch.cp("old_sg")
    cp_new_sg = scratch.cp("new_sg")
    # n_moved comes from propose_batch_kernel output, not the meta column
    cp_nm = scratch.cp("n_moved_out")
    cp_ci = scratch.cp("col_chain")[gs:gs + 1]
    cp_bs = scratch.cp("col_bead")[gs:gs + 1]

    M = scratch.M
    B = 1
    block_dim = 256
    block = (block_dim, 1, 1)
    grid = (B, 1, 1)
    smem = block_dim * 4
    inv_box = 1.0 / box

    kernel = ck.get_delta_e_kernel()
    torch_stream = torch.cuda.current_stream(ca_t.device)
    cp_stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)
    with cp_stream:
        kernel(grid, block, (
            cp_ca, cp_sg,
            cp_old_ca, cp_new_ca, cp_old_sg, cp_new_sg,
            cp_nm, cp_ci, cp_bs, cp_de,
            np.int32(M), np.int32(B),
            np.int32(n_chains), np.int32(N),
            np.float32(box), np.float32(inv_box),
            np.float32(r_rep_sq), np.float32(r_max_sq),
            np.float32(rep_e), np.float32(contact_e),
            np.int32(min_caca), np.int32(min_casg), np.int32(min_sgsg),
        ), shared_mem=smem)

    return scratch.delta_e


def accept_segment_b1(scratch, ca_t, sg_t, gs: int, N: int, kBT: float):
    """Apply Metropolis accept + writeback on the device.

    Reads delta_e and the propose-output buffers, decides acceptance with
    a pre-drawn random, mutates state.ca/state.sg in place if accepted,
    and atomically updates the per-move-type counters in scratch.
    """
    import cupy as cp
    from . import cuda_kernels as ck

    cp_ca = scratch.cp_state_ca(ca_t)
    cp_sg = scratch.cp_state_sg(sg_t)
    cp_new_ca = scratch.cp("new_ca")
    cp_new_sg = scratch.cp("new_sg")
    cp_de = scratch.cp("delta_e")
    cp_nm = scratch.cp("n_moved_out")
    cp_mt = scratch.cp("move_type")
    cp_ci = scratch.cp("col_chain")[gs:gs + 1]
    cp_bs = scratch.cp("col_bead")[gs:gs + 1]
    cp_ra = scratch.cp("rand_accept")[gs:gs + 1]
    cp_att = scratch.cp("attempted_counts")
    cp_acc = scratch.cp("accepted_counts")

    M = scratch.M
    B = 1
    block = (32, 1, 1)
    grid = (B, 1, 1)
    inv_kBT = 1.0 / float(kBT)

    kernel = ck.get_accept_kernel()
    torch_stream = torch.cuda.current_stream(ca_t.device)
    cp_stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)
    with cp_stream:
        kernel(grid, block, (
            cp_ca, cp_sg, cp_new_ca, cp_new_sg,
            cp_de, cp_nm, cp_mt, cp_ci, cp_bs, cp_ra,
            cp_att, cp_acc,
            np.int32(M), np.int32(B), np.int32(N),
            np.float32(inv_kBT),
        ))


def mc_sweep_kernel(scratch, ca_t, sg_t, n_segs: int,
                    N: int, n_chains: int, box: float,
                    max_angle: float, kBT: float,
                    r_rep_sq, r_max_sq, rep_e, contact_e,
                    min_caca, min_casg, min_sgsg) -> float:
    """Single-launch mega-kernel: full sweep (all segments) in one block.

    Eliminates per-segment Python dispatch and per-segment kernel launches.
    The block iterates the precomputed permutation in scratch.perm_buf,
    processing each segment through propose -> delta_e -> accept ->
    writeback in shared memory + global state mutation.
    """
    import cupy as cp
    from . import cuda_kernels as ck

    cp_ca = scratch.cp_state_ca(ca_t)
    cp_sg = scratch.cp_state_sg(sg_t)
    cp_meta = scratch.cp("table_full")
    cp_perm = scratch.cp("perm_buf")[:n_segs]
    cp_ra = scratch.cp("rand_buf")
    cp_rc = scratch.cp("rand_accept")
    cp_att = scratch.cp("attempted_counts")
    cp_acc = scratch.cp("accepted_counts")

    M = scratch.M
    block_dim = 256
    block = (block_dim, 1, 1)
    grid = (1, 1, 1)

    # Shared mem layout (see cuda_kernels.py mc_sweep_kernel header):
    #   5*M*3 + 8 (positions+headers) + block_dim (partial_de) + 1 (sh_accept)
    smem_floats = 5 * M * 3 + 8 + block_dim + 1
    smem_bytes = smem_floats * 4

    inv_box = 1.0 / box
    half_box = 0.5 * box
    inv_kBT = 1.0 / float(kBT)

    kernel = ck.get_mc_sweep_kernel()
    if smem_bytes > 49152:
        kernel.max_dynamic_shared_size_bytes = smem_bytes
    torch_stream = torch.cuda.current_stream(ca_t.device)
    cp_stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)
    evt_start = cp.cuda.Event()
    evt_end = cp.cuda.Event()
    with cp_stream:
        evt_start.record()
        kernel(grid, block, (
            cp_ca, cp_sg,
            cp_meta, cp_perm, cp_ra, cp_rc,
            cp_att, cp_acc,
            np.int32(n_segs), np.int32(N), np.int32(M), np.int32(n_chains),
            np.float32(box), np.float32(inv_box), np.float32(half_box),
            np.float32(max_angle), np.float32(inv_kBT),
            np.float32(r_rep_sq), np.float32(r_max_sq),
            np.float32(rep_e), np.float32(contact_e),
            np.int32(min_caca), np.int32(min_casg), np.int32(min_sgsg),
        ), shared_mem=smem_bytes)
        evt_end.record()
    evt_end.synchronize()
    return float(cp.cuda.get_elapsed_time(evt_start, evt_end))


def correction_matrix_torch(
        old_ca, new_ca, old_sg, new_sg, n_moved,
        chain_idx, bead_start, N, box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg,
):
    """Dense BxB upper-triangular rank-1 correction.

    Computed pair-wise; each surviving (i, j) does the 4 cross-energies
    (E00 + E11 - E01 - E10) over the 2*nm_i x 2*nm_j moved beads.
    """
    B = old_ca.shape[0]
    device = old_ca.device
    dtype = old_ca.dtype
    inv_box = 1.0 / box
    corr = torch.zeros(B, B, dtype=dtype, device=device)

    def _pair_e(p1, p2, mask):
        d = p2.unsqueeze(0) - p1.unsqueeze(1)
        d -= box * torch.round(d * inv_box)
        r2 = (d * d).sum(dim=-1)
        return (_zone_e(r2, r_rep_sq, r_max_sq, rep_e, contact_e) * mask).sum()

    for i in range(B):
        nm_i = int(n_moved[i].item())
        if nm_i == 0:
            continue
        ci = int(chain_idx[i].item())
        ms_i = int(bead_start[i].item())
        for j in range(i + 1, B):
            nm_j = int(n_moved[j].item())
            if nm_j == 0:
                continue
            cj = int(chain_idx[j].item())
            ms_j = int(bead_start[j].item())
            ri = torch.arange(ms_i, ms_i + nm_i, device=device)
            rj = torch.arange(ms_j, ms_j + nm_j, device=device)
            seq = (ri.unsqueeze(1) - rj.unsqueeze(0)).abs()
            same_chain = (ci == cj)
            skip_caca = (seq < min_caca) if same_chain else torch.zeros_like(seq, dtype=torch.bool)
            skip_casg = (seq < min_casg) if same_chain else torch.zeros_like(seq, dtype=torch.bool)
            skip_sgsg = (seq < min_sgsg) if same_chain else torch.zeros_like(seq, dtype=torch.bool)
            keep_caca = (~skip_caca).to(dtype)
            keep_casg = (~skip_casg).to(dtype)
            keep_sgsg = (~skip_sgsg).to(dtype)

            # Cα-Cα
            e00 = _pair_e(old_ca[i, :nm_i], old_ca[j, :nm_j], keep_caca)
            e01 = _pair_e(old_ca[i, :nm_i], new_ca[j, :nm_j], keep_caca)
            e10 = _pair_e(new_ca[i, :nm_i], old_ca[j, :nm_j], keep_caca)
            e11 = _pair_e(new_ca[i, :nm_i], new_ca[j, :nm_j], keep_caca)
            # Cα(i) vs SG(j)
            e00 += _pair_e(old_ca[i, :nm_i], old_sg[j, :nm_j], keep_casg)
            e01 += _pair_e(old_ca[i, :nm_i], new_sg[j, :nm_j], keep_casg)
            e10 += _pair_e(new_ca[i, :nm_i], old_sg[j, :nm_j], keep_casg)
            e11 += _pair_e(new_ca[i, :nm_i], new_sg[j, :nm_j], keep_casg)
            # SG(i) vs Cα(j)
            e00 += _pair_e(old_sg[i, :nm_i], old_ca[j, :nm_j], keep_casg)
            e01 += _pair_e(old_sg[i, :nm_i], new_ca[j, :nm_j], keep_casg)
            e10 += _pair_e(new_sg[i, :nm_i], old_ca[j, :nm_j], keep_casg)
            e11 += _pair_e(new_sg[i, :nm_i], new_ca[j, :nm_j], keep_casg)
            # SG-SG
            e00 += _pair_e(old_sg[i, :nm_i], old_sg[j, :nm_j], keep_sgsg)
            e01 += _pair_e(old_sg[i, :nm_i], new_sg[j, :nm_j], keep_sgsg)
            e10 += _pair_e(new_sg[i, :nm_i], old_sg[j, :nm_j], keep_sgsg)
            e11 += _pair_e(new_sg[i, :nm_i], new_sg[j, :nm_j], keep_sgsg)
            corr[i, j] = (e11 - e01) - (e10 - e00)
    return corr
