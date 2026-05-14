"""Fully batched delta-E — torch CPU tensors, no per-segment Python loop.

Computes delta_E for ALL B proposals in one fused tensor op.

Wrapped with torch.compile so TorchInductor can fuse the broadcast subtract /
wrap / square / sum / mask / reduce chain into one C++/OpenMP kernel and
eliminate intermediate-tensor allocations. The eager core is kept as
`_batch_delta_e_torch_fused_eager` for fallback when compile is unavailable
(e.g., no MSVC on Windows). On first call the compile triggers and may
spend a few seconds JIT-ing; subsequent calls hit the cached kernel.
"""

from __future__ import annotations

import os
import torch

_USE_COMPILE = os.environ.get("ROUSE_NO_TORCH_COMPILE", "0") != "1"


def _zone_e(r2: torch.Tensor, r_rep_sq: float, r_max_sq: float,
            rep_e: float, contact_e: float) -> torch.Tensor:
    e = torch.where(r2 < r_rep_sq, torch.full_like(r2, rep_e), torch.zeros_like(r2))
    if contact_e != 0.0:
        e = torch.where((r2 >= r_rep_sq) & (r2 < r_max_sq),
                        torch.full_like(r2, contact_e), e)
    return e


def _pair_r2_mic(moved: torch.Tensor, ref: torch.Tensor,
                 box: float, inv_box: float) -> torch.Tensor:
    """Squared minimum-image distances between every moved bead and every
    reference bead.

    moved: (P, 3); ref: (Q, 3); returns: (P, Q) float.
    """
    d = ref.unsqueeze(0) - moved.unsqueeze(1)        # (P, Q, 3)
    d = d - box * torch.round(d * inv_box)
    return (d * d).sum(dim=-1)                       # (P, Q)


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
      old_ca, new_ca, old_sg, new_sg : (B, M, 3)   torch float
      n_moved_out, chain_idx, bead_start : (B,)    torch long
      ca_full, sg_full              : (n_chains, N, 3) torch float
    """
    B, M, _ = old_ca.shape
    device = old_ca.device
    dtype = old_ca.dtype
    inv_box = 1.0 / box
    n_chains = ca_full.shape[0]
    NK = n_chains * N

    # Lane validity (B, M) → flat (B*M,)
    k_range = torch.arange(M, device=device).unsqueeze(0)        # (1, M)
    lane_valid = k_range < n_moved_out.unsqueeze(1)              # (B, M) bool
    lane_valid_f = lane_valid.reshape(B * M)                     # (B*M,)

    # Per-lane chain + residue id
    bead_resid = bead_start.unsqueeze(1) + k_range               # (B, M)
    moved_chain = chain_idx.unsqueeze(1).expand(B, M)            # (B, M)
    moved_chain_f = moved_chain.reshape(B * M)                   # (B*M,)
    moved_resid_f = bead_resid.reshape(B * M)                    # (B*M,)

    # Flatten lane axis
    moved_ca_old = old_ca.reshape(B * M, 3)
    moved_ca_new = new_ca.reshape(B * M, 3)
    moved_sg_old = old_sg.reshape(B * M, 3)
    moved_sg_new = new_sg.reshape(B * M, 3)

    # Reference state flat (NK, 3)
    ca_flat = ca_full.reshape(NK, 3)
    sg_flat = sg_full.reshape(NK, 3)
    ref_idx = torch.arange(NK, device=device)
    ref_chain = ref_idx // N                                     # (NK,)
    ref_resid = ref_idx % N

    # Same-chain mask (B*M, NK)
    same_chain = moved_chain_f.unsqueeze(1) == ref_chain.unsqueeze(0)
    seq_diff = (moved_resid_f.unsqueeze(1) - ref_resid.unsqueeze(0)).abs()

    # Self-exclusion: a moved lane's bead must not interact with any bead
    # in its own moved span (chain_idx[b], resid in [bead_start[b], bead_start[b]+n_moved[b])).
    seg_start_f = (bead_start.unsqueeze(1).expand(B, M).reshape(B * M)).unsqueeze(1)
    seg_end_f = ((bead_start + n_moved_out).unsqueeze(1).expand(B, M).reshape(B * M)).unsqueeze(1)
    in_moved_span = (ref_resid.unsqueeze(0) >= seg_start_f) & \
                    (ref_resid.unsqueeze(0) < seg_end_f)
    self_excl = same_chain & in_moved_span                       # (B*M, NK)

    skip_caca = same_chain & (seq_diff < min_caca)
    skip_casg = same_chain & (seq_diff < min_casg)
    skip_sgsg = same_chain & (seq_diff < min_sgsg)

    keep_caca = (~(skip_caca | self_excl)) & lane_valid_f.unsqueeze(1)
    keep_casg = (~(skip_casg | self_excl)) & lane_valid_f.unsqueeze(1)
    keep_sgsg = (~(skip_sgsg | self_excl)) & lane_valid_f.unsqueeze(1)
    m_caca = keep_caca.to(dtype)
    m_casg = keep_casg.to(dtype)
    m_sgsg = keep_sgsg.to(dtype)

    # Concatenate moved + ref across pair types to do ONE big cdist instead of 4.
    # Moved set:  [Ca_old | Ca_new | SG_old | SG_new]   ->  (4*B*M, 3)
    # Ref set:    [Ca | SG]                              ->  (2*NK, 3)
    # Big r2 has shape (4*B*M, 2*NK); we slice it back to the 8 sub-matrices.
    moved_all = torch.cat([moved_ca_old, moved_ca_new, moved_sg_old, moved_sg_new], dim=0)
    ref_all = torch.cat([ca_flat, sg_flat], dim=0)
    P = moved_all.shape[0]                              # 4 * B*M
    Q = ref_all.shape[0]                                # 2 * NK
    d = ref_all.unsqueeze(0) - moved_all.unsqueeze(1)   # (P, Q, 3)
    d = d - box * torch.round(d * inv_box)
    r2_big = (d * d).sum(dim=-1)                        # (P, Q)
    e_big = _zone_e(r2_big, r_rep_sq, r_max_sq, rep_e, contact_e)   # (P, Q)

    BM = B * M
    # Row blocks: [0..BM)=Ca_old moved, [BM..2BM)=Ca_new, [2BM..3BM)=SG_old, [3BM..4BM)=SG_new
    # Col blocks: [0..NK)=Ca_ref, [NK..2NK)=SG_ref
    e_caca_old = (e_big[:BM, :NK] * m_caca).sum(dim=1)
    e_casg_a_old = (e_big[:BM, NK:] * m_casg).sum(dim=1)
    e_caca_new = (e_big[BM:2 * BM, :NK] * m_caca).sum(dim=1)
    e_casg_a_new = (e_big[BM:2 * BM, NK:] * m_casg).sum(dim=1)
    e_casg_b_old = (e_big[2 * BM:3 * BM, :NK] * m_casg).sum(dim=1)
    e_sgsg_old = (e_big[2 * BM:3 * BM, NK:] * m_sgsg).sum(dim=1)
    e_casg_b_new = (e_big[3 * BM:4 * BM, :NK] * m_casg).sum(dim=1)
    e_sgsg_new = (e_big[3 * BM:4 * BM, NK:] * m_sgsg).sum(dim=1)

    e_lane_old = e_caca_old + e_casg_a_old + e_casg_b_old + e_sgsg_old   # (B*M,)
    e_lane_new = e_caca_new + e_casg_a_new + e_casg_b_new + e_sgsg_new

    delta_lane = e_lane_new - e_lane_old                                  # (B*M,)
    delta_e = delta_lane.reshape(B, M).sum(dim=1)                         # (B,)
    return delta_e


# Wrap with torch.compile / TorchScript for graph-level optimisation.
# On Windows the inductor backend needs MSVC's `cl` which may not be
# present — we suppress dynamo errors and the call falls back to eager.
# Selection order, from most-aggressive to fallback:
#   1. torch.compile (mode=reduce-overhead, backend=inductor) — best CPU
#      fusion, requires cl on Windows.
#   2. torch.compile (backend=aot_eager) — graph trace only, no external
#      compiler; modest overhead reduction.
#   3. torch.jit.script — TorchScript fallback.
#   4. eager — original Python loop dispatch.
# Configure via ROUSE_TORCH_COMPILE_MODE = inductor | aot_eager | script | eager
#                                                       (default: inductor).
_COMPILE_MODE = os.environ.get("ROUSE_TORCH_COMPILE_MODE", "inductor").lower()

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
    # Otherwise try torch.compile with the selected backend.
    try:
        if _COMPILE_MODE == "aot_eager":
            return torch.compile(fn, backend="aot_eager", fullgraph=False, dynamic=False)
        if _COMPILE_MODE == "max-autotune":
            return torch.compile(fn, mode="max-autotune", fullgraph=False, dynamic=False)
        return torch.compile(fn, mode="reduce-overhead", fullgraph=False, dynamic=False)
    except Exception:
        return fn


batch_delta_e_torch_fused = _resolve_compiled()

# Backward-compat alias for the C17/C18 checklist synthetic tests that
# import `batch_delta_e_torch` directly from this module.
batch_delta_e_torch = batch_delta_e_torch_fused
