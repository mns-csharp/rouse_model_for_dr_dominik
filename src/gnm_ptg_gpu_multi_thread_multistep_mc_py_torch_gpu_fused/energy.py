"""Fully batched delta-E — torch CUDA tensors, GPU-resident, no Python loop.

Computes delta_E for ALL B proposals in one fused tensor op. State stays on
the GPU; no host transfer happens here. Wrapped with torch.compile so the
broadcast subtract / wrap / square / sum / mask / reduce chain fuses into a
single CUDA kernel and dispatch overhead goes from O(B*ops) to O(1).

The eager core is kept as `_batch_delta_e_torch_fused_eager` for fallback
when compile is unavailable (graceful degrade, no behavioural change).

Public symbol: `batch_delta_e_torch_fused` (compiled by default).
Override via `ROUSE_TORCH_COMPILE_MODE = inductor | aot_eager | eager`.
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


def _batch_delta_e_torch_fused_eager(
        old_ca, new_ca, old_sg, new_sg, n_moved_out,
        chain_idx, bead_start,
        ca_full, sg_full,
        N, box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg,
):
    """Fused delta_E across all B proposals.

    Shapes:
      old_ca, new_ca, old_sg, new_sg : (B, M, 3)   torch float on CUDA
      n_moved_out, chain_idx, bead_start : (B,)    torch long on CUDA
      ca_full, sg_full              : (n_chains, N, 3) torch float on CUDA
    Returns delta_e: (B,) on CUDA.
    """
    B, M, _ = old_ca.shape
    device = old_ca.device
    dtype = old_ca.dtype
    inv_box = 1.0 / box
    n_chains = ca_full.shape[0]
    NK = n_chains * N

    k_range = torch.arange(M, device=device).unsqueeze(0)
    lane_valid = k_range < n_moved_out.unsqueeze(1)
    lane_valid_f = lane_valid.reshape(B * M)

    bead_resid = bead_start.unsqueeze(1) + k_range
    moved_chain = chain_idx.unsqueeze(1).expand(B, M)
    moved_chain_f = moved_chain.reshape(B * M)
    moved_resid_f = bead_resid.reshape(B * M)

    moved_ca_old = old_ca.reshape(B * M, 3)
    moved_ca_new = new_ca.reshape(B * M, 3)
    moved_sg_old = old_sg.reshape(B * M, 3)
    moved_sg_new = new_sg.reshape(B * M, 3)

    ca_flat = ca_full.reshape(NK, 3)
    sg_flat = sg_full.reshape(NK, 3)
    ref_idx = torch.arange(NK, device=device)
    ref_chain = ref_idx // N
    ref_resid = ref_idx % N

    same_chain = moved_chain_f.unsqueeze(1) == ref_chain.unsqueeze(0)
    seq_diff = (moved_resid_f.unsqueeze(1) - ref_resid.unsqueeze(0)).abs()

    seg_start_f = (bead_start.unsqueeze(1).expand(B, M).reshape(B * M)).unsqueeze(1)
    seg_end_f = ((bead_start + n_moved_out).unsqueeze(1).expand(B, M).reshape(B * M)).unsqueeze(1)
    in_moved_span = (ref_resid.unsqueeze(0) >= seg_start_f) & \
                    (ref_resid.unsqueeze(0) < seg_end_f)
    self_excl = same_chain & in_moved_span

    skip_caca = same_chain & (seq_diff < min_caca)
    skip_casg = same_chain & (seq_diff < min_casg)
    skip_sgsg = same_chain & (seq_diff < min_sgsg)

    keep_caca = (~(skip_caca | self_excl)) & lane_valid_f.unsqueeze(1)
    keep_casg = (~(skip_casg | self_excl)) & lane_valid_f.unsqueeze(1)
    keep_sgsg = (~(skip_sgsg | self_excl)) & lane_valid_f.unsqueeze(1)
    m_caca = keep_caca.to(dtype)
    m_casg = keep_casg.to(dtype)
    m_sgsg = keep_sgsg.to(dtype)

    # Concatenate moved + ref to do ONE big distance computation. The GPU
    # sees this as a single launch instead of 8 separate ones.
    moved_all = torch.cat([moved_ca_old, moved_ca_new, moved_sg_old, moved_sg_new], dim=0)
    ref_all = torch.cat([ca_flat, sg_flat], dim=0)
    d = ref_all.unsqueeze(0) - moved_all.unsqueeze(1)
    d = d - box * torch.round(d * inv_box)
    r2_big = (d * d).sum(dim=-1)
    e_big = _zone_e(r2_big, r_rep_sq, r_max_sq, rep_e, contact_e)

    BM = B * M
    e_caca_old = (e_big[:BM, :NK] * m_caca).sum(dim=1)
    e_casg_a_old = (e_big[:BM, NK:] * m_casg).sum(dim=1)
    e_caca_new = (e_big[BM:2 * BM, :NK] * m_caca).sum(dim=1)
    e_casg_a_new = (e_big[BM:2 * BM, NK:] * m_casg).sum(dim=1)
    e_casg_b_old = (e_big[2 * BM:3 * BM, :NK] * m_casg).sum(dim=1)
    e_sgsg_old = (e_big[2 * BM:3 * BM, NK:] * m_sgsg).sum(dim=1)
    e_casg_b_new = (e_big[3 * BM:4 * BM, :NK] * m_casg).sum(dim=1)
    e_sgsg_new = (e_big[3 * BM:4 * BM, NK:] * m_sgsg).sum(dim=1)

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
    fn = _batch_delta_e_torch_fused_eager
    if not _USE_COMPILE or _COMPILE_MODE == "eager":
        return fn
    if _COMPILE_MODE == "script":
        try:
            return torch.jit.script(fn)
        except Exception:
            return fn
    try:
        if _COMPILE_MODE == "aot_eager":
            return torch.compile(fn, backend="aot_eager", fullgraph=False, dynamic=False)
        if _COMPILE_MODE == "max-autotune":
            return torch.compile(fn, mode="max-autotune", fullgraph=False, dynamic=False)
        return torch.compile(fn, mode="reduce-overhead", fullgraph=False, dynamic=False)
    except Exception:
        return fn


batch_delta_e_torch_fused = _resolve_compiled()

# Back-compat alias so checklist tests and any sibling imports still resolve.
batch_delta_e_torch = batch_delta_e_torch_fused


def _correction_matrix_torch_fused_eager(
        old_ca, new_ca, old_sg, new_sg, n_moved,
        chain_idx, bead_start, N, box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg,
):
    """Fully vectorised rank-1 EMM correction matrix.

    Builds the (B, B) correction matrix as one batched operation. For each
    ordered pair (i, j) with i<j:
        corr[i, j] = (e_new_new + e_old_old) - (e_new_old + e_old_new)
                   = sum_{pair_type} sum_{k,l} sgn[k_i,l_j] * E[...]

    where the moved beads of each proposal are stacked along the M axis:
    moved_ca = cat([old_ca, new_ca], dim=1) → (B, 2M, 3). The "sign" tensor
    sgn[k_i, l_j] is +1 when k_i and l_j live in the same half (both old or
    both new) and -1 otherwise. Same for moved_sg.

    Skip masks come from sequence-separation rules of each pair-type
    (CA-CA, CA-SG, SG-CA, SG-SG) and the lane-validity mask (k_i < n_moved[i]).

    All work stays on the GPU; output corr is (B, B) on the same device.
    """
    B, M, _ = old_ca.shape
    device = old_ca.device
    dtype = old_ca.dtype
    inv_box = 1.0 / box

    # Stack moved beads: first M lanes are "old", next M are "new".
    moved_ca = torch.cat([old_ca, new_ca], dim=1)  # (B, 2M, 3)
    moved_sg = torch.cat([old_sg, new_sg], dim=1)

    # Per-lane validity in the 2M axis: index k_local = k % M maps to the
    # underlying moved-bead index; lane k is valid iff k_local < n_moved[i].
    k_axis = torch.arange(2 * M, device=device)
    k_local = k_axis % M                                            # (2M,)
    lane_valid = k_local.unsqueeze(0) < n_moved.unsqueeze(1)        # (B, 2M)

    # Per-lane residue offset within the segment is k_local; absolute residue
    # is bead_start[i] + k_local. Sequence-separation between (i, k_i) and
    # (j, l_j) is |resid_i - resid_j|, only meaningful when same chain.
    resid = bead_start.unsqueeze(1) + k_local.unsqueeze(0)          # (B, 2M)

    # Sign tensor (2M, 2M): +1 if both halves match (old-old or new-new), -1 else.
    half_idx = (k_axis >= M).to(torch.long)                         # (2M,) 0 or 1
    sgn = torch.where(half_idx.unsqueeze(0) == half_idx.unsqueeze(1),
                      torch.ones((2 * M, 2 * M), device=device, dtype=dtype),
                      -torch.ones((2 * M, 2 * M), device=device, dtype=dtype))

    # Same-chain mask (B, B) and seq_diff (B, B, 2M, 2M).
    same_chain = chain_idx.unsqueeze(1) == chain_idx.unsqueeze(0)   # (B, B)
    seq_diff = (resid.unsqueeze(1).unsqueeze(3)
                - resid.unsqueeze(0).unsqueeze(2)).abs()            # (B, B, 2M, 2M)

    skip_caca = same_chain.unsqueeze(-1).unsqueeze(-1) & (seq_diff < min_caca)
    skip_casg = same_chain.unsqueeze(-1).unsqueeze(-1) & (seq_diff < min_casg)
    skip_sgsg = same_chain.unsqueeze(-1).unsqueeze(-1) & (seq_diff < min_sgsg)

    # Self-pair (i == j) must not contribute to corr (corr is i<j only, but
    # we'd zero-out the diagonal at the end anyway). Lane validity:
    lane_pair = lane_valid.unsqueeze(1).unsqueeze(3) & \
                lane_valid.unsqueeze(0).unsqueeze(2)                # (B, B, 2M, 2M)

    keep_caca = (~skip_caca) & lane_pair
    keep_casg = (~skip_casg) & lane_pair
    keep_sgsg = (~skip_sgsg) & lane_pair

    def _pair_zone(moved_a, moved_b):
        # moved_a, moved_b : (B, 2M, 3); returns (B, B, 2M, 2M) zone energies.
        # diff[i, j, k, l, :] = moved_b[j, l, :] - moved_a[i, k, :]
        d = moved_b.unsqueeze(0).unsqueeze(2) - moved_a.unsqueeze(1).unsqueeze(3)
        d = d - box * torch.round(d * inv_box)
        r2 = (d * d).sum(dim=-1)                                    # (B, B, 2M, 2M)
        e = torch.where(r2 < r_rep_sq,
                        torch.full_like(r2, rep_e),
                        torch.zeros_like(r2))
        if contact_e != 0.0:
            e = torch.where((r2 >= r_rep_sq) & (r2 < r_max_sq),
                            torch.full_like(r2, contact_e), e)
        return e

    e_caca = _pair_zone(moved_ca, moved_ca)
    e_casg = _pair_zone(moved_ca, moved_sg)
    e_sgca = _pair_zone(moved_sg, moved_ca)
    e_sgsg = _pair_zone(moved_sg, moved_sg)

    # corr[i, j] = sum over (k_i, l_j) of sgn[k_i, l_j] * (
    #     e_caca * keep_caca + e_casg * keep_casg + e_sgca * keep_casg + e_sgsg * keep_sgsg
    # )
    sgn_b = sgn.unsqueeze(0).unsqueeze(0)                           # (1,1,2M,2M)
    weighted = (e_caca * keep_caca.to(dtype) * sgn_b
                + e_casg * keep_casg.to(dtype) * sgn_b
                + e_sgca * keep_casg.to(dtype) * sgn_b
                + e_sgsg * keep_sgsg.to(dtype) * sgn_b)
    corr_full = weighted.sum(dim=(-1, -2))                          # (B, B)

    # Zero out diagonal (no self-correction); upper triangle is what
    # _fused_accept reads (j > i), but we keep symmetric form for clarity.
    eye = torch.eye(B, device=device, dtype=torch.bool)
    corr_full = corr_full.masked_fill(eye, 0.0)
    return corr_full


correction_matrix_torch_fused = _correction_matrix_torch_fused_eager
correction_matrix_torch = correction_matrix_torch_fused

