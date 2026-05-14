"""Fused-batched multistep MC sweep — torch CUDA tensors, GPU-resident.

Migacz-EMM (Exact Marginalisation of Mutations) multistep MC. Per batch:
  1. Build the (B, 6) meta tensor from the round permutation; B <= batch_size.
  2. ONE propose_batch_torch call -> (B, M, 3) old/new ca/sg.
  3. ONE batch_delta_e_torch_fused call -> (B,) delta_e (vs reference state).
  4. ONE correction_matrix_torch_fused call -> (B, B) cross-proposal corr.
  5. Host-side sequential-with-correction Metropolis: each acceptance at i
     updates downstream delta_e[j] += corr[i, j] before j's accept decision.
  6. Scatter accepted moves back into GPU state.

The host-side accept loop is O(B) Python iterations per batch — trivial vs
the GPU work. State remains on GPU throughout; transfers only for the (B,)
delta_e and (B, B) correction at accept time.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .chain import ChainState, SegmentInfo
from .config import SimConfig
from .energy import batch_delta_e_torch_fused, correction_matrix_torch_fused
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

    def acceptance(self, mtype: str) -> float:
        a = self.attempted[mtype]
        return self.accepted[mtype] / a if a > 0 else 0.0


def _fused_accept(delta_e: np.ndarray, correction: np.ndarray,
                  n_moved: np.ndarray, kBT: float, uniforms: np.ndarray):
    """Sequential Metropolis with rank-1 update — runs on host."""
    B = delta_e.shape[0]
    accepted = np.zeros(B, dtype=np.bool_)
    u_idx = 0
    for i in range(B):
        if n_moved[i] == 0:
            continue
        de = float(delta_e[i])
        if de <= 0.0:
            prob = 1.0
        else:
            exponent = -de / kBT
            if exponent <= -745.0:
                prob = 0.0
            elif exponent >= 709.0:
                prob = 1.0
            else:
                prob = math.exp(exponent)
        if uniforms[u_idx] < prob:
            accepted[i] = True
            for j in range(i + 1, B):
                if n_moved[j] == 0:
                    continue
                c = float(correction[i, j])
                if c != 0.0:
                    delta_e[j] = float(delta_e[j]) + c
        u_idx += 1
    return accepted


def _build_round_permutation(seg_info: SegmentInfo, n_chains: int, rng):
    """Build a shuffled segment permutation where each round (slot k) groups
    segments that share the same in-chain position so anchors don't collide.
    Returns (perm, boundaries) where boundaries[r:r+1] marks the slice of
    perm holding round r's segments.
    """
    total = seg_info.total_segments
    spc = seg_info.segs_per_chain
    table = seg_info.table
    buckets = [[] for _ in range(n_chains)]
    for gs in range(total):
        buckets[int(table[gs, SegmentInfo.META_CHAIN_IDX])].append(gs)
    for bucket in buckets:
        rng.shuffle(bucket)
    rounds = []
    for k in range(spc):
        round_segs = [b[k] for b in buckets if k < len(b)]
        rng.shuffle(round_segs)
        rounds.append(round_segs)
    perm = []
    boundaries = [0]
    for r in rounds:
        perm.extend(r)
        boundaries.append(len(perm))
    return np.asarray(perm, dtype=np.int64), boundaries


def perform_sweep(state, cfg: SimConfig, stats: SimulationStats,
                  rng: np.random.Generator) -> None:
    """GPU-resident fused multistep sweep."""
    seg_info = state.segments
    N = cfg.N
    box = cfg.box_size
    kBT = cfg.kBT
    M = max(int(seg_info.table[:, 2].max()), 1)
    n_chains = state.cfg.n_chains
    device = state.device
    dtype = state.dtype

    ca_t = state.ca
    sg_t = state.sg
    table_full = seg_info.table

    perm, boundaries = _build_round_permutation(seg_info, n_chains, rng)
    B_max = min(int(getattr(cfg, "batch_size", 256)), n_chains)

    batch_ranges = []
    for ri in range(len(boundaries) - 1):
        for bs in range(boundaries[ri], boundaries[ri + 1], B_max):
            be = min(bs + B_max, boundaries[ri + 1])
            batch_ranges.append((bs, be))

    for bs, be in batch_ranges:
        B = be - bs
        if B == 0:
            continue
        meta_np = table_full[perm[bs:be]]
        meta_t = torch.from_numpy(meta_np).to(device)
        angle_unif = torch.rand(B, device=device, dtype=dtype)

        old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type = propose_batch_torch(
            ca_t, sg_t, meta_t, angle_unif, M, N, box, cfg.max_angle_hinge,
        )

        delta_e = batch_delta_e_torch_fused(
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            meta_t[:, 0].long(), meta_t[:, 1].long(),
            ca_t, sg_t,
            N, box,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )

        corr = correction_matrix_torch_fused(
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            meta_t[:, 0].long(), meta_t[:, 1].long(), N, box,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )

        diff = new_ca - old_ca
        diff = diff - box * torch.round(diff / box)
        disps = diff.norm(dim=-1)
        lane_mask = (torch.arange(M, device=device).unsqueeze(0)
                     < n_moved_out.unsqueeze(1))
        disps = disps * lane_mask.to(disps.dtype)

        de_np = delta_e.detach().cpu().numpy().astype(np.float64).copy()
        corr_np = corr.detach().cpu().numpy().astype(np.float64).copy()
        nm_np = n_moved_out.detach().cpu().numpy().astype(np.int64)
        mt_np = move_type.detach().cpu().numpy().astype(np.int64)
        u_np = rng.random(B).astype(np.float64)

        accepted = _fused_accept(de_np, corr_np, nm_np, kBT, u_np)

        CHECKLIST_COUNTERS["ev_calls"] += int((nm_np > 0).sum())
        d_max = float(disps.max().item()) if B > 0 else 0.0
        if d_max > CHECKLIST_COUNTERS["max_disp"]:
            CHECKLIST_COUNTERS["max_disp"] = d_max
        CHECKLIST_COUNTERS["disp_violations"] += int(
            ((disps > cfg.l0 * 1.05) & lane_mask).sum().item())

        for i in range(B):
            if nm_np[i] == 0:
                stats.record(_MOVE_NAMES[int(mt_np[i])], False)
                continue
            stats.record(_MOVE_NAMES[int(mt_np[i])], bool(accepted[i]))

        acc_mask = torch.from_numpy(accepted & (nm_np > 0)).to(device)
        if bool(acc_mask.any().item()):
            accept_idx = acc_mask.nonzero(as_tuple=False).squeeze(-1)
            ci_t = meta_t[accept_idx, 0].long()
            ms_t = meta_t[accept_idx, 1].long()
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

    return None
