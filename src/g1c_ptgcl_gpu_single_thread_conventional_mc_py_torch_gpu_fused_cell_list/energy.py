"""Cell-list delta-E — torch CUDA tensors, GPU-resident.

Same fused-batch design as `g1c_ptg/energy.py` but the (4*B*M, 2*NK) all-pairs
reduction is replaced with a (4*B*M, max_neighbors) cell-list-bounded
reduction. Memory now scales O(B * M * max_neighbors) instead of O(B * M * NK)
— at K=10000 N=100, max_neighbors~256 vs NK=1,000,000, a 3900x memory drop.

Per moved lane (B*M, total 4 lanes per proposal for ca_old/new + sg_old/new):
  - The candidate index tensor (P, max_neighbors) is supplied by the caller
    (`cell_list.gather_candidates(...)`).
  - We gather the reference positions, chain_idx, and resid for each candidate.
  - We mask out:
      * lane_valid (k < n_moved[b])
      * pad (candidate idx == -1)
      * self-pair (candidate is in the moved bead's own span)
      * sequence-separation skip (CA-CA<min_caca, CA-SG<min_casg, SG-SG<min_sgsg)
  - MIC distance + zone energy + reduce-sum -> (B,) delta_e.

Wrapped with `torch.compile` (inductor) as before.
"""

from __future__ import annotations

import os
import torch

_USE_COMPILE = os.environ.get("ROUSE_NO_TORCH_COMPILE", "0") != "1"
_COMPILE_MODE = os.environ.get("ROUSE_TORCH_COMPILE_MODE", "inductor").lower()


def _zone_e(r2: torch.Tensor, r_rep_sq: float, r_max_sq: float,
            rep_e: float, contact_e: float) -> torch.Tensor:
    e = torch.where(r2 < r_rep_sq, torch.full_like(r2, rep_e), torch.zeros_like(r2))
    if contact_e != 0.0:
        e = torch.where((r2 >= r_rep_sq) & (r2 < r_max_sq),
                        torch.full_like(r2, contact_e), e)
    return e


def _batch_delta_e_cell_list_eager(
        old_ca, new_ca, old_sg, new_sg, n_moved_out,
        chain_idx, bead_start,
        ca_full, sg_full,
        cand_ca_old, cand_ca_new, cand_sg_old, cand_sg_new,
        N, box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg,
):
    """Cell-list delta_E across all B proposals.

    Shapes:
      old_ca, new_ca, old_sg, new_sg : (B, M, 3)
      n_moved_out, chain_idx, bead_start : (B,)
      ca_full, sg_full              : (n_chains, N, 3)
      cand_*                        : (B*M, max_neighbors); -1 padding.
    """
    B, M, _ = old_ca.shape
    device = old_ca.device
    dtype = old_ca.dtype
    inv_box = 1.0 / box
    n_chains = ca_full.shape[0]
    NK = n_chains * N
    BM = B * M

    k_range = torch.arange(M, device=device).unsqueeze(0)
    lane_valid = k_range < n_moved_out.unsqueeze(1)
    lane_valid_f = lane_valid.reshape(BM)

    bead_resid = bead_start.unsqueeze(1) + k_range
    moved_chain_f = chain_idx.unsqueeze(1).expand(B, M).reshape(BM)
    moved_resid_f = bead_resid.reshape(BM)
    seg_start_f = bead_start.unsqueeze(1).expand(B, M).reshape(BM)
    seg_end_f = (bead_start + n_moved_out).unsqueeze(1).expand(B, M).reshape(BM)

    ref_idx_all = torch.arange(NK, device=device)
    ref_chain_all = ref_idx_all // N
    ref_resid_all = ref_idx_all % N

    ca_flat = ca_full.reshape(NK, 3)
    sg_flat = sg_full.reshape(NK, 3)

    moved_ca_old = old_ca.reshape(BM, 3)
    moved_ca_new = new_ca.reshape(BM, 3)
    moved_sg_old = old_sg.reshape(BM, 3)
    moved_sg_new = new_sg.reshape(BM, 3)

    def _pair_block(moved_xyz, cand_idx, ref_flat, min_sep):
        valid_cand = cand_idx >= 0
        cand_idx_clamped = cand_idx.clamp(min=0)
        ref_xyz = ref_flat[cand_idx_clamped]
        ref_chain = ref_chain_all[cand_idx_clamped]
        ref_resid = ref_resid_all[cand_idx_clamped]

        same_chain = moved_chain_f.unsqueeze(1) == ref_chain
        seq_diff = (moved_resid_f.unsqueeze(1) - ref_resid).abs()
        skip_seq = same_chain & (seq_diff < min_sep)
        in_moved_span = same_chain & (ref_resid >= seg_start_f.unsqueeze(1)) \
                                   & (ref_resid < seg_end_f.unsqueeze(1))
        keep = valid_cand & lane_valid_f.unsqueeze(1) & (~skip_seq) & (~in_moved_span)

        d = ref_xyz - moved_xyz.unsqueeze(1)
        d = d - box * torch.round(d * inv_box)
        r2 = (d * d).sum(dim=-1)
        e = _zone_e(r2, r_rep_sq, r_max_sq, rep_e, contact_e)
        return (e * keep.to(dtype)).sum(dim=1)

    e_caca_old = _pair_block(moved_ca_old, cand_ca_old, ca_flat, min_caca)
    e_caca_new = _pair_block(moved_ca_new, cand_ca_new, ca_flat, min_caca)
    e_casg_a_old = _pair_block(moved_ca_old, cand_ca_old, sg_flat, min_casg)
    e_casg_a_new = _pair_block(moved_ca_new, cand_ca_new, sg_flat, min_casg)
    e_casg_b_old = _pair_block(moved_sg_old, cand_sg_old, ca_flat, min_casg)
    e_casg_b_new = _pair_block(moved_sg_new, cand_sg_new, ca_flat, min_casg)
    e_sgsg_old = _pair_block(moved_sg_old, cand_sg_old, sg_flat, min_sgsg)
    e_sgsg_new = _pair_block(moved_sg_new, cand_sg_new, sg_flat, min_sgsg)

    e_lane_old = e_caca_old + e_casg_a_old + e_casg_b_old + e_sgsg_old
    e_lane_new = e_caca_new + e_casg_a_new + e_casg_b_new + e_sgsg_new
    delta_lane = e_lane_new - e_lane_old
    delta_e = delta_lane.reshape(B, M).sum(dim=1)
    return delta_e


try:
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True
except Exception:
    pass


def _resolve_compiled():
    fn = _batch_delta_e_cell_list_eager
    if not _USE_COMPILE or _COMPILE_MODE == "eager":
        return fn
    try:
        if _COMPILE_MODE == "aot_eager":
            return torch.compile(fn, backend="aot_eager", fullgraph=False, dynamic=False)
        if _COMPILE_MODE == "max-autotune":
            return torch.compile(fn, mode="max-autotune", fullgraph=False, dynamic=False)
        return torch.compile(fn, mode="reduce-overhead", fullgraph=False, dynamic=False)
    except Exception:
        return fn


batch_delta_e_torch_cell_list_fused = _resolve_compiled()


def batch_delta_e_torch(
        old_ca, new_ca, old_sg, new_sg, n_moved_out,
        chain_idx, bead_start,
        ca_full, sg_full,
        N, box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg,
):
    """Legacy all-pairs signature for the synthetic checklist test (B=1,
    no-op move, used to verify delta_E(no-move) == 0). Builds a candidate
    index list covering ALL beads (equivalent to all-pairs), then dispatches
    to the cell-list kernel. Not used in the hot path.
    """
    device = old_ca.device
    n_chains = ca_full.shape[0]
    NK = n_chains * N
    B, M, _ = old_ca.shape
    BM = B * M
    all_idx = torch.arange(NK, device=device).unsqueeze(0).expand(BM, NK).contiguous()
    return batch_delta_e_torch_cell_list_fused(
        old_ca, new_ca, old_sg, new_sg, n_moved_out,
        chain_idx, bead_start, ca_full, sg_full,
        all_idx, all_idx, all_idx, all_idx,
        N, box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg,
    )


# Aliases.
batch_delta_e_torch_fused = batch_delta_e_torch
