"""Segmented conventional MC sweep — torch CPU tensors, single-thread.

Per-segment sequential Metropolis. No batching, no rank-1 correction.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .chain import ChainState, SegmentInfo
from .config import SimConfig
from .energy import batch_delta_e_torch
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


def perform_sweep(state, cfg: SimConfig, stats: SimulationStats,
                  rng: np.random.Generator) -> None:
    seg_info = state.segments
    table_t = torch.from_numpy(seg_info.table)
    N = cfg.N
    box = cfg.box_size
    kBT = cfg.kBT

    order = np.arange(seg_info.total_segments, dtype=np.int64)
    rng.shuffle(order)

    M = max(seg_info.max_moved_static, 1)
    for gs in order:
        ca_t = torch.from_numpy(state.ca)
        sg_t = torch.from_numpy(state.sg)
        meta_t = table_t[int(gs):int(gs) + 1]
        rand_t = torch.from_numpy(rng.random(1).astype(np.float64))

        old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type = propose_batch_torch(
            ca_t, sg_t, meta_t, rand_t, M, N, box, cfg.max_angle_hinge,
        )

        nm = int(n_moved_out[0].item())
        mtype = _MOVE_NAMES[int(move_type[0].item())]
        if nm == 0:
            stats.record(mtype, False)
            continue

        delta_e = batch_delta_e_torch(
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            meta_t[:, 0].long(), meta_t[:, 1].long(),
            ca_t, sg_t,
            N, box,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )
        CHECKLIST_COUNTERS["ev_calls"] += 1
        de = float(delta_e[0].item())

        diff = (new_ca[0, :nm] - old_ca[0, :nm])
        diff = diff - box * torch.round(diff / box)
        disps = diff.norm(dim=-1)
        d_max = float(disps.max())
        if d_max > CHECKLIST_COUNTERS["max_disp"]:
            CHECKLIST_COUNTERS["max_disp"] = d_max
        n_viol = int((disps > cfg.l0 * 1.05).sum().item())
        CHECKLIST_COUNTERS["disp_violations"] += n_viol
        if de <= 0.0:
            prob = 1.0
        else:
            exponent = -de / kBT
            prob = 0.0 if exponent <= -745.0 else (1.0 if exponent >= 709.0
                                                   else math.exp(exponent))
        accept = rng.random() < prob
        stats.record(mtype, accept)
        if accept:
            ci = int(meta_t[0, 0].item())
            ms = int(meta_t[0, 1].item())
            state.ca[ci, ms:ms + nm, :] = new_ca[0, :nm].numpy()
            state.sg[ci, ms:ms + nm, :] = new_sg[0, :nm].numpy()
