"""Segmented Migacz multistep MC sweep — torch CPU tensors, single-thread."""

from __future__ import annotations

import math

import numpy as np
import torch

from .chain import ChainState, SegmentInfo
from .config import SimConfig
from .energy import batch_delta_e_torch, correction_matrix_torch
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
    seg_info = state.segments
    table_t = torch.from_numpy(seg_info.table)
    N = cfg.N
    n_chains = cfg.n_chains
    box = cfg.box_size
    kBT = cfg.kBT

    # ChainState in this app holds numpy arrays. Wrap them as torch tensors.
    ca_t = torch.from_numpy(state.ca)
    sg_t = torch.from_numpy(state.sg)

    perm, boundaries = _build_round_permutation(seg_info, n_chains, rng)
    B_max = min(cfg.batch_size, n_chains)
    max_moved = max(seg_info.max_moved_static, 1)

    batch_ranges = []
    for ri in range(len(boundaries) - 1):
        for bs in range(boundaries[ri], boundaries[ri + 1], B_max):
            be = min(bs + B_max, boundaries[ri + 1])
            batch_ranges.append((bs, be))

    for bs, be in batch_ranges:
        B = be - bs
        meta_t = table_t[torch.from_numpy(perm[bs:be])]
        rand_t = torch.from_numpy(rng.random(B).astype(np.float64))

        old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type = propose_batch_torch(
            ca_t, sg_t, meta_t, rand_t, max_moved, N,
            cfg.box_size, cfg.max_angle_hinge,
        )

        delta_e = batch_delta_e_torch(
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            meta_t[:, 0].long(), meta_t[:, 1].long(),
            ca_t, sg_t,
            N, cfg.box_size,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )
        corr = correction_matrix_torch(
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            meta_t[:, 0].long(), meta_t[:, 1].long(), N, cfg.box_size,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )

        # Host-side accept loop.
        de_np = delta_e.numpy().copy()
        corr_np = corr.numpy().copy()
        nm_np = n_moved_out.numpy().astype(np.int64)
        u_np = rng.random(B).astype(np.float64)
        accepted = _fused_accept(de_np, corr_np, nm_np, kBT, u_np)

        CHECKLIST_COUNTERS["ev_calls"] += int(B)
        diffs = (new_ca - old_ca)
        diffs = diffs - cfg.box_size * torch.round(diffs / cfg.box_size)
        disps_all = diffs.norm(dim=-1)
        if int(disps_all.numel()) > 0:
            d_max = float(disps_all.max())
            if d_max > CHECKLIST_COUNTERS["max_disp"]:
                CHECKLIST_COUNTERS["max_disp"] = d_max
            n_viol = int((disps_all > cfg.l0 * 1.05).sum().item())
            CHECKLIST_COUNTERS["disp_violations"] += n_viol

        # Apply accepted moves to numpy state and refresh torch views.
        for i in range(B):
            nm = int(nm_np[i])
            mtype = _MOVE_NAMES[int(move_type[i].item())]
            stats.record(mtype, bool(accepted[i]))
            if nm == 0 or not accepted[i]:
                continue
            ci = int(meta_t[i, 0].item())
            ms = int(meta_t[i, 1].item())
            state.ca[ci, ms:ms + nm, :] = new_ca[i, :nm].numpy()
            state.sg[ci, ms:ms + nm, :] = new_sg[i, :nm].numpy()
        # Re-bind torch views to mutated numpy state for the next batch.
        ca_t = torch.from_numpy(state.ca)
        sg_t = torch.from_numpy(state.sg)
