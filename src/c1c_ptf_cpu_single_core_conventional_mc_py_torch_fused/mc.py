"""Fused-batched MC sweep — torch CPU tensors, single-thread.

Per sweep:
  1. Shuffle the (total_segs,) global segment-index permutation.
  2. Build the (B,6) meta table for all segments in shuffled order (B = total_segs).
  3. Single batched propose_batch_torch call -> (B, M, 3) old/new ca/sg.
  4. Single batched batch_delta_e_torch_fused call -> (B,) delta_e.
  5. Vectorised parallel-Gibbs Metropolis accept (no Python loop).
  6. Scatter accepted moves back into state.ca / state.sg (vectorised).

No `for gs in order` loop, no per-call `.item()`, no per-call
`torch.from_numpy(state.ca)`. State tensors live in torch from sim init.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .chain import ChainState, SegmentInfo
from .config import SimConfig
from .energy import batch_delta_e_torch_fused
from .proposer import propose_batch_torch, MTYPE_HINGE, MTYPE_N_TAIL, MTYPE_C_TAIL


_MOVE_NAMES = ("hinge", "n_tail", "c_tail")


CHECKLIST_COUNTERS = {
    "ev_calls": 0,
    "max_disp": 0.0,
    "disp_violations": 0,
}


def _reset_counters():
    CHECKLIST_COUNTERS["ev_calls"] = 0
    CHECKLIST_COUNTERS["max_disp"] = 0.0
    CHECKLIST_COUNTERS["disp_violations"] = 0


class SimulationStats:
    def __init__(self):
        self.attempted = {"hinge": 0, "n_tail": 0, "c_tail": 0}
        self.accepted = {"hinge": 0, "n_tail": 0, "c_tail": 0}

    def record(self, mtype: str, accepted: bool):
        self.attempted[mtype] += 1
        if accepted:
            self.accepted[mtype] += 1

    def add_batch(self, attempted_arr, accepted_arr) -> None:
        for i, name in enumerate(_MOVE_NAMES):
            self.attempted[name] += int(attempted_arr[i])
            self.accepted[name] += int(accepted_arr[i])

    def acceptance(self, mtype: str) -> float:
        a = self.attempted[mtype]
        return self.accepted[mtype] / a if a > 0 else 0.0


def _run_round(meta_t, ca_t, sg_t, n_chains_in_round, M, N, box, kBT, cfg,
               stats, rng):
    """One round of the sweep: one segment per chain, all chains in parallel.

    All segments in a round share the same `bead_start` within their chain
    (the round number * residues_per_segment), so their anchors lie strictly
    outside any other segment in the round (the other segments live in
    different chains). Parallel-Gibbs is therefore exact for intra-chain
    geometry, with only the inter-chain energy-couplig approximation
    remaining.
    """
    B = meta_t.shape[0]
    angle_unif = torch.from_numpy(rng.random(B).astype(np.float64))
    metro_unif = torch.from_numpy(rng.random(B).astype(np.float64))

    old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type = propose_batch_torch(
        ca_t, sg_t, meta_t, angle_unif, M, N, box, cfg.max_angle_hinge,
    )
    valid_lane = (n_moved_out > 0)

    delta_e = batch_delta_e_torch_fused(
        old_ca, new_ca, old_sg, new_sg, n_moved_out,
        meta_t[:, 0].long(), meta_t[:, 1].long(),
        ca_t, sg_t,
        N, box,
        cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
        cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
    )

    CHECKLIST_COUNTERS["ev_calls"] += int(valid_lane.sum().item())

    # Displacement tracker (per round).
    diff = new_ca - old_ca
    diff = diff - box * torch.round(diff / box)
    disps = diff.norm(dim=-1)
    lane_mask = (torch.arange(M).unsqueeze(0) < n_moved_out.unsqueeze(1))
    disps = disps * lane_mask.to(disps.dtype)
    if B > 0:
        d_max = float(disps.max().item())
        if d_max > CHECKLIST_COUNTERS["max_disp"]:
            CHECKLIST_COUNTERS["max_disp"] = d_max
    CHECKLIST_COUNTERS["disp_violations"] += int(
        ((disps > cfg.l0 * 1.05) & lane_mask).sum().item()
    )

    # Vectorised parallel-Gibbs Metropolis accept.
    exponent = -delta_e / kBT
    exp_clamped = exponent.clamp(min=-745.0, max=0.0)
    accept_prob = torch.where(delta_e <= 0,
                              torch.ones_like(delta_e),
                              torch.exp(exp_clamped))
    accept = (metro_unif < accept_prob) & valid_lane

    # Per-move-type bookkeeping.
    attempted_per_type = torch.zeros(3, dtype=torch.long)
    accepted_per_type = torch.zeros(3, dtype=torch.long)
    for t in range(3):
        mask_t = (move_type == t) & valid_lane
        attempted_per_type[t] = int(mask_t.sum().item())
        accepted_per_type[t] = int((mask_t & accept).sum().item())
    stats.add_batch(attempted_per_type.numpy(), accepted_per_type.numpy())

    if accept.any():
        accept_idx = accept.nonzero(as_tuple=False).squeeze(-1)
        ci_t = meta_t[accept_idx, 0].long()
        ms_t = meta_t[accept_idx, 1].long()
        nm_t = n_moved_out[accept_idx].long()
        k_idx = torch.arange(M).unsqueeze(0)
        res_idx = ms_t.unsqueeze(1) + k_idx                      # (A, M)
        lane_ok = k_idx < nm_t.unsqueeze(1)                      # (A, M) bool
        chain_flat = ci_t.unsqueeze(1).expand_as(res_idx).reshape(-1)
        res_flat = res_idx.reshape(-1)
        new_ca_flat = new_ca[accept_idx].reshape(-1, 3)
        new_sg_flat = new_sg[accept_idx].reshape(-1, 3)
        lane_flat = lane_ok.reshape(-1)                          # (A*M,) bool

        # Filter to valid lanes only — never scatter padding zeros.
        # Without this, tail segments (nm < M) collide on res_idx after
        # clamping and overwrite live residues with padding.
        valid_pos = lane_flat.nonzero(as_tuple=False).squeeze(-1)
        chain_sel = chain_flat[valid_pos]
        res_sel = res_flat[valid_pos]
        ca_sel = new_ca_flat[valid_pos]
        sg_sel = new_sg_flat[valid_pos]

        ca_t.index_put_((chain_sel, res_sel), ca_sel, accumulate=False)
        sg_t.index_put_((chain_sel, res_sel), sg_sel, accumulate=False)


def perform_sweep(state, cfg: SimConfig, stats: SimulationStats,
                  rng: np.random.Generator) -> None:
    """Round-based fused sweep: one segment per chain per round, with rounds
    iterated in shuffled order.

    Within a round, all segments live on different chains, so their
    intra-chain anchors do not collide (parallel-Gibbs is exact for the
    Calpha bond geometry). Inter-chain energy coupling is the only remaining
    approximation (delta_E for each round uses the state at the start of
    that round, not after sibling accepts within the same round).

    Replaces the original per-segment Python loop (~N*K iterations) with
    `n_segments_per_chain` rounds (typically 4 at N=25, residues_per_segment=8).
    """
    seg_info = state.segments
    N = cfg.N
    box = cfg.box_size
    kBT = cfg.kBT
    # The static `max_moved_static` overestimates by `N - seg_size` when
    # `segs_per_chain > 1`; we pin to the true max n_moved across the table
    # so the proposer + fused delta-E don't waste (M × NK) memory on padding.
    M = max(int(seg_info.table[:, 2].max()), 1)
    n_chains = state.cfg.n_chains
    segs_per_chain = seg_info.segs_per_chain

    # Persistent torch views over the numpy state (zero-copy).
    ca_t = torch.from_numpy(state.ca)
    sg_t = torch.from_numpy(state.sg)
    table_full = seg_info.table  # numpy (total_segs, 6)
    # `table_full[c * segs_per_chain + s]` is the global segment for (c, s).
    # Build per-round meta tensors once (chain ordering shuffled per sweep).
    chain_order = np.arange(n_chains, dtype=np.int64)
    rng.shuffle(chain_order)
    round_order = np.arange(segs_per_chain, dtype=np.int64)
    rng.shuffle(round_order)

    for s in round_order:
        rows = chain_order * segs_per_chain + int(s)             # (n_chains,)
        meta_round_np = table_full[rows]
        meta_round_t = torch.from_numpy(meta_round_np)
        _run_round(meta_round_t, ca_t, sg_t, n_chains, M, N, box, kBT, cfg,
                   stats, rng)

    return None
