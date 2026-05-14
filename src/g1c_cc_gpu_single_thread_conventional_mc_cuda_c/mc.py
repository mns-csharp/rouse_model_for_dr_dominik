"""Segmented conventional MC sweep -- single mega-kernel per sweep.

The mc_sweep_kernel processes all segments in one CUDA launch with
sequential per-segment steps (propose -> delta_e -> accept -> writeback)
inside one thread block. Eliminates per-segment Python dispatch and
~390k kernel launches per 1000-sweep run.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .chain import ChainState
from .config import SimConfig
from .energy import mc_sweep_kernel
from .proposer import MTYPE_HINGE, MTYPE_N_TAIL, MTYPE_C_TAIL


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


def perform_sweep(state, cfg: SimConfig, stats: SimulationStats,
                  rng: np.random.Generator, scratch) -> float:
    """Run one full sweep on the GPU. Returns device-side kernel ms."""
    seg_info = state.segments
    N = cfg.N
    n_chains = cfg.n_chains
    box = cfg.box_size
    kBT = cfg.kBT

    total_segs = seg_info.total_segments
    perm_host = np.arange(total_segs, dtype=np.int64)
    rng.shuffle(perm_host)
    scratch.perm_buf.copy_(torch.from_numpy(perm_host), non_blocking=True)

    scratch.fill_rand(rng)
    scratch.reset_counters()

    kernel_ms = mc_sweep_kernel(
        scratch, state.ca, state.sg, total_segs,
        N, n_chains, box,
        cfg.max_angle_hinge, kBT,
        cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
        cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
    )

    attempted_h = scratch.attempted_counts.cpu().numpy()
    accepted_h = scratch.accepted_counts.cpu().numpy()
    stats.add_batch(attempted_h, accepted_h)
    CHECKLIST_COUNTERS["ev_calls"] += int(attempted_h.sum())
    return kernel_ms
