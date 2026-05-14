"""Fused-batched MC sweep — torch CUDA tensors, GPU-resident, single-thread.

Algorithm per sweep:
  Repeat `n_segments_per_chain` rounds (typically 4 at N=25 / seg_size=8):
    1. Build the (n_chains, 6) meta tensor for this round on the GPU.
       Each chain contributes exactly one segment, all at the same in-chain
       slot index, so segment anchors in the round do not collide across
       chains. Parallel-Gibbs is exact for intra-chain geometry.
    2. One propose_batch_torch call -> (n_chains, M, 3) old/new ca/sg.
    3. One batch_delta_e_torch_fused call -> (n_chains,) delta_e.
    4. Vectorised parallel-Gibbs Metropolis accept on the GPU.
    5. Scatter accepted moves into the GPU state tensors via index_put_.

No Python `for` loop over segments. No `.item()` inside the sweep before
accept-scatter. No host transfer of state. Random numbers are produced
on the GPU directly (torch.rand) so the only host->device synchronisation
per sweep is shuffling the chain/round order numpy permutations.
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


def _run_round(meta_round_t, ca_t, sg_t, M, N, box, kBT, cfg, stats, device):
    """One parallel-Gibbs round on the GPU. All tensors are CUDA-resident."""
    B = meta_round_t.shape[0]
    dtype = ca_t.dtype
    angle_unif = torch.rand(B, device=device, dtype=dtype)
    metro_unif = torch.rand(B, device=device, dtype=dtype)

    old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type = propose_batch_torch(
        ca_t, sg_t, meta_round_t, angle_unif, M, N, box, cfg.max_angle_hinge,
    )
    valid_lane = (n_moved_out > 0)

    delta_e = batch_delta_e_torch_fused(
        old_ca, new_ca, old_sg, new_sg, n_moved_out,
        meta_round_t[:, 0].long(), meta_round_t[:, 1].long(),
        ca_t, sg_t,
        N, box,
        cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
        cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
    )

    # Displacement audit (kept on GPU, single sync at end of round).
    diff = new_ca - old_ca
    diff = diff - box * torch.round(diff / box)
    disps = diff.norm(dim=-1)
    lane_mask = (torch.arange(M, device=device).unsqueeze(0)
                 < n_moved_out.unsqueeze(1))
    disps = disps * lane_mask.to(disps.dtype)

    exponent = -delta_e / kBT
    exp_clamped = exponent.clamp(min=-745.0, max=0.0)
    accept_prob = torch.where(delta_e <= 0,
                              torch.ones_like(delta_e),
                              torch.exp(exp_clamped))
    accept = (metro_unif < accept_prob) & valid_lane

    # Per-move-type stats: reduce on GPU then bring back 6 scalars/round.
    mt_h = (move_type == MTYPE_HINGE) & valid_lane
    mt_n = (move_type == MTYPE_N_TAIL) & valid_lane
    mt_c = (move_type == MTYPE_C_TAIL) & valid_lane
    att = torch.stack([mt_h.sum(), mt_n.sum(), mt_c.sum()]).to(torch.long)
    acc = torch.stack([(mt_h & accept).sum(),
                       (mt_n & accept).sum(),
                       (mt_c & accept).sum()]).to(torch.long)
    ev_calls = int(valid_lane.sum().item())
    d_max = float(disps.max().item()) if B > 0 else 0.0
    n_viol = int(((disps > cfg.l0 * 1.05) & lane_mask).sum().item())

    CHECKLIST_COUNTERS["ev_calls"] += ev_calls
    if d_max > CHECKLIST_COUNTERS["max_disp"]:
        CHECKLIST_COUNTERS["max_disp"] = d_max
    CHECKLIST_COUNTERS["disp_violations"] += n_viol
    stats.add_batch(att.cpu().numpy(), acc.cpu().numpy())

    if bool(accept.any().item()):
        accept_idx = accept.nonzero(as_tuple=False).squeeze(-1)
        ci_t = meta_round_t[accept_idx, 0].long()
        ms_t = meta_round_t[accept_idx, 1].long()
        nm_t = n_moved_out[accept_idx].long()
        k_idx = torch.arange(M, device=device).unsqueeze(0)
        res_idx = ms_t.unsqueeze(1) + k_idx
        lane_ok = k_idx < nm_t.unsqueeze(1)
        chain_flat = ci_t.unsqueeze(1).expand_as(res_idx).reshape(-1)
        res_flat = res_idx.reshape(-1)
        new_ca_flat = new_ca[accept_idx].reshape(-1, 3)
        new_sg_flat = new_sg[accept_idx].reshape(-1, 3)
        lane_flat = lane_ok.reshape(-1)

        valid_pos = lane_flat.nonzero(as_tuple=False).squeeze(-1)
        chain_sel = chain_flat[valid_pos]
        res_sel = res_flat[valid_pos]
        ca_sel = new_ca_flat[valid_pos]
        sg_sel = new_sg_flat[valid_pos]

        ca_t.index_put_((chain_sel, res_sel), ca_sel, accumulate=False)
        sg_t.index_put_((chain_sel, res_sel), sg_sel, accumulate=False)


def perform_sweep(state, cfg: SimConfig, stats: SimulationStats,
                  rng: np.random.Generator) -> None:
    """GPU-resident round-based fused sweep.

    `n_segments_per_chain` rounds per sweep, one (chain, slot) per row of the
    round meta tensor — built once on the host (numpy index gymnastics),
    pushed to the GPU once per round. Everything else stays on the GPU.
    """
    seg_info = state.segments
    N = cfg.N
    box = cfg.box_size
    kBT = cfg.kBT
    M = max(int(seg_info.table[:, 2].max()), 1)
    n_chains = state.cfg.n_chains
    segs_per_chain = seg_info.segs_per_chain
    device = state.device

    ca_t = state.ca
    sg_t = state.sg
    table_full = seg_info.table  # numpy (total_segs, 6)

    chain_order = np.arange(n_chains, dtype=np.int64)
    rng.shuffle(chain_order)
    round_order = np.arange(segs_per_chain, dtype=np.int64)
    rng.shuffle(round_order)

    for s in round_order:
        rows = chain_order * segs_per_chain + int(s)
        meta_round_np = table_full[rows]
        meta_round_t = torch.from_numpy(meta_round_np).to(device)
        _run_round(meta_round_t, ca_t, sg_t, M, N, box, kBT, cfg, stats, device)

    return None
