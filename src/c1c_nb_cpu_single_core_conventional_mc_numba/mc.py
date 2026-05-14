"""Segmented conventional (sequential) Metropolis MC sweep — single-core CPU.

Each proposal rotates a whole chain *segment* (one segment of residues_per_segment
consecutive Calpha+SG pairs) as a rigid body. Per Migacz literature,
"conventional" means:
  - One segment proposal at a time, accept/reject before the next is generated.
  - No rank-1 EMM correction matrix.
  - Cell list MUST be rebuilt before every proposal so the energy reflects
    every accepted move (`context_rouse_benchmark.txt:817-821`).
"""

from __future__ import annotations

import math

import numpy as np

from .chain import ChainState, SegmentInfo
from .config import SimConfig
from .energy import build_cell_list, proposal_delta_e
from .proposer import propose_batch, MTYPE_HINGE, MTYPE_N_TAIL, MTYPE_C_TAIL


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


def perform_sweep(state: ChainState, cfg: SimConfig,
                  stats: SimulationStats, rng: np.random.Generator) -> None:
    seg_info = state.segments
    table = seg_info.table
    N = cfg.N
    box = cfg.box_size
    inv_box = 1.0 / box
    half_box = cfg.half_box
    max_angle = cfg.max_angle_hinge
    l0 = cfg.l0
    kBT = cfg.kBT
    rep_e = cfg.repulsive_energy
    contact_e = cfg.contact_energy
    r_rep_sq = cfg.r_rep_sq
    r_max_sq = cfg.r_max_sq
    cap_inner = cfg.cap_inner_hinge
    cap_tail = cfg.cap_tail
    min_caca = cfg.min_seq_caca
    min_casg = cfg.min_seq_casg
    min_sgsg = cfg.min_seq_sgsg

    # Single random shuffle of the segment indices each sweep.
    order = np.arange(seg_info.total_segments, dtype=np.int64)
    rng.shuffle(order)

    # Cell-list grid is per-sweep static (cells), but counts/order rebuild each
    # proposal because state changes after every accept.
    nc, cell_size, neighbor_offsets = state.ns.make_cell_grid(cfg.r_max)
    inv_cs = 1.0 / cell_size

    # Reuse 1-row scratch buffers (B = 1).
    chain_idx_arr = np.zeros(1, dtype=np.int64)
    bead_start_arr = np.zeros(1, dtype=np.int64)
    n_moved_arr_in = np.zeros(1, dtype=np.int64)
    seg_type_arr = np.zeros(1, dtype=np.int64)
    anchor_a_arr = np.zeros(1, dtype=np.int64)
    anchor_b_arr = np.zeros(1, dtype=np.int64)
    n_moved_arr_out = np.zeros(1, dtype=np.int64)
    move_type_arr = np.zeros(1, dtype=np.int64)
    angle_unif = np.zeros(1, dtype=np.float64)

    M = max(seg_info.max_moved_static, 1)
    old_ca = np.zeros((1, M, 3), dtype=np.float64)
    new_ca = np.zeros((1, M, 3), dtype=np.float64)
    old_sg = np.zeros((1, M, 3), dtype=np.float64)
    new_sg = np.zeros((1, M, 3), dtype=np.float64)

    for gs in order:
        chain_idx_arr[0] = table[gs, SegmentInfo.META_CHAIN_IDX]
        bead_start_arr[0] = table[gs, SegmentInfo.META_BEAD_START]
        n_moved_arr_in[0] = table[gs, SegmentInfo.META_N_MOVED]
        seg_type_arr[0] = table[gs, SegmentInfo.META_SEG_TYPE]
        anchor_a_arr[0] = table[gs, SegmentInfo.META_ANCHOR_A]
        anchor_b_arr[0] = table[gs, SegmentInfo.META_ANCHOR_B]
        angle_unif[0] = rng.random()

        propose_batch(
            state.ca, state.sg,
            chain_idx_arr, bead_start_arr, n_moved_arr_in,
            seg_type_arr, anchor_a_arr, anchor_b_arr,
            angle_unif,
            old_ca, new_ca, old_sg, new_sg,
            n_moved_arr_out, move_type_arr,
            N, box, inv_box, half_box,
            max_angle, l0, cap_inner, cap_tail,
        )

        nm = int(n_moved_arr_out[0])
        mtype = _MOVE_NAMES[int(move_type_arr[0])]
        if nm == 0:
            stats.record(mtype, False)
            continue

        # Rebuild cell list with the *current* state (key conventional-MC requirement).
        flat_pos = state.get_flat_beads()
        sorted_order, cell_starts, cell_counts = build_cell_list(
            flat_pos, nc, inv_cs, half_box)

        delta_e = proposal_delta_e(
            old_ca[0], new_ca[0], old_sg[0], new_sg[0], nm,
            int(chain_idx_arr[0]), int(bead_start_arr[0]), N,
            flat_pos,
            nc, inv_cs, half_box, neighbor_offsets,
            sorted_order, cell_starts, cell_counts,
            box, inv_box,
            r_rep_sq, r_max_sq, rep_e, contact_e,
            min_caca, min_casg, min_sgsg,
        )
        CHECKLIST_COUNTERS["ev_calls"] += 1

        # Track max bead displacement vs. its old position; flag violations.
        for k in range(nm):
            dx = new_ca[0, k, 0] - old_ca[0, k, 0]
            dy = new_ca[0, k, 1] - old_ca[0, k, 1]
            dz = new_ca[0, k, 2] - old_ca[0, k, 2]
            dx -= box * round(dx * inv_box)
            dy -= box * round(dy * inv_box)
            dz -= box * round(dz * inv_box)
            disp = math.sqrt(dx * dx + dy * dy + dz * dz)
            if disp > CHECKLIST_COUNTERS["max_disp"]:
                CHECKLIST_COUNTERS["max_disp"] = disp
            if disp > l0 * 1.05:
                CHECKLIST_COUNTERS["disp_violations"] += 1

        if delta_e <= 0.0:
            prob = 1.0
        else:
            exponent = -delta_e / kBT
            if exponent <= -745.0:
                prob = 0.0
            elif exponent >= 709.0:
                prob = 1.0
            else:
                prob = math.exp(exponent)
        accept = rng.random() < prob
        stats.record(mtype, accept)

        if accept:
            ci = int(chain_idx_arr[0])
            ms = int(bead_start_arr[0])
            for k in range(nm):
                state.ca[ci, ms + k, 0] = new_ca[0, k, 0]
                state.ca[ci, ms + k, 1] = new_ca[0, k, 1]
                state.ca[ci, ms + k, 2] = new_ca[0, k, 2]
                state.sg[ci, ms + k, 0] = new_sg[0, k, 0]
                state.sg[ci, ms + k, 1] = new_sg[0, k, 1]
                state.sg[ci, ms + k, 2] = new_sg[0, k, 2]
