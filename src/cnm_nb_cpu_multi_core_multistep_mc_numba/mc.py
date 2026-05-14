"""Segmented Migacz multistep MC sweep — multicore CPU.

Each MC proposal rotates a whole chain *segment* (residues_per_segment
consecutive Calpha+SG pairs) as a rigid body. Sparse rank-1 EMM correction
is computed *between segment proposals* (only for pairs whose bounding
spheres overlap).

Phases:
  1. Round-based segment ordering (host).
  2. Parallel batch proposal (Numba prange over B).
  3. Parallel batch delta-E (Numba prange over B).
  4. Parallel sparse rank-1 EMM correction (Numba prange over upper-triangular pairs).
  5. Sequential accept + rank-1 update (causal; runs serial on the main thread).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .chain import ChainState, SegmentInfo
from .config import SimConfig
from .energy import build_cell_list_par, batch_delta_e_par, correction_matrix_sparse
from .proposer import propose_batch_par


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


@njit(cache=True, fastmath=False)
def _fused_accept_and_correct(delta_e, correction, n_moved, kBT, uniforms):
    B = delta_e.shape[0]
    accepted = np.zeros(B, dtype=np.bool_)
    u_idx = 0
    for i in range(B):
        if n_moved[i] == 0:
            continue
        de = delta_e[i]
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
                if n_moved[j] == 0: continue
                c = correction[i, j]
                if c != 0.0:
                    delta_e[j] += c
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


def perform_sweep(state: ChainState, cfg: SimConfig,
                  stats: SimulationStats, rng: np.random.Generator,
                  cell_rebuild_interval: int = 8) -> None:
    seg_info = state.segments
    table = seg_info.table
    N = cfg.N
    n_chains = cfg.n_chains
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
    r_max = cfg.r_max
    cap_inner = cfg.cap_inner_hinge
    cap_tail = cfg.cap_tail
    min_caca = cfg.min_seq_caca
    min_casg = cfg.min_seq_casg
    min_sgsg = cfg.min_seq_sgsg

    if int(getattr(cfg, "batch_size", 1)) == 1:
        perm = np.arange(seg_info.total_segments, dtype=np.int64)
        rng.shuffle(perm)
        boundaries = list(range(0, len(perm) + 1))
    else:
        perm, boundaries = _build_round_permutation(seg_info, n_chains, rng)

    nc, cell_size, neighbor_offsets = state.ns.make_cell_grid(cfg.r_max)
    inv_cs = 1.0 / cell_size

    max_moved = max(seg_info.max_moved_static, 1)
    B_max = min(cfg.batch_size, n_chains)

    chain_idx_arr = np.zeros(B_max, dtype=np.int64)
    bead_start_arr = np.zeros(B_max, dtype=np.int64)
    n_moved_arr_in = np.zeros(B_max, dtype=np.int64)
    seg_type_arr = np.zeros(B_max, dtype=np.int64)
    anchor_a_arr = np.zeros(B_max, dtype=np.int64)
    anchor_b_arr = np.zeros(B_max, dtype=np.int64)
    n_moved_arr_out = np.zeros(B_max, dtype=np.int64)
    move_type_arr = np.zeros(B_max, dtype=np.int64)
    angle_unif = np.zeros(B_max, dtype=np.float64)
    accept_unif = np.zeros(B_max, dtype=np.float64)
    old_ca = np.zeros((B_max, max_moved, 3), dtype=np.float64)
    new_ca = np.zeros((B_max, max_moved, 3), dtype=np.float64)
    old_sg = np.zeros((B_max, max_moved, 3), dtype=np.float64)
    new_sg = np.zeros((B_max, max_moved, 3), dtype=np.float64)

    batch_ranges = []
    for ri in range(len(boundaries) - 1):
        r_start = boundaries[ri]
        r_end = boundaries[ri + 1]
        for bs in range(r_start, r_end, B_max):
            be = min(bs + B_max, r_end)
            batch_ranges.append((bs, be))

    flat_pos = state.get_flat_beads()
    sorted_order, cell_starts, cell_counts = build_cell_list_par(
        flat_pos, nc, inv_cs, half_box)

    for b_idx, (bs, be) in enumerate(batch_ranges):
        B = be - bs
        if b_idx > 0 and b_idx % cell_rebuild_interval == 0:
            flat_pos = state.get_flat_beads()
            sorted_order, cell_starts, cell_counts = build_cell_list_par(
                flat_pos, nc, inv_cs, half_box)

        for slot in range(B):
            gs = int(perm[bs + slot])
            chain_idx_arr[slot] = table[gs, SegmentInfo.META_CHAIN_IDX]
            bead_start_arr[slot] = table[gs, SegmentInfo.META_BEAD_START]
            n_moved_arr_in[slot] = table[gs, SegmentInfo.META_N_MOVED]
            seg_type_arr[slot] = table[gs, SegmentInfo.META_SEG_TYPE]
            anchor_a_arr[slot] = table[gs, SegmentInfo.META_ANCHOR_A]
            anchor_b_arr[slot] = table[gs, SegmentInfo.META_ANCHOR_B]

        angle_unif[:B] = rng.random(B)
        accept_unif[:B] = rng.random(B)

        propose_batch_par(
            state.ca, state.sg,
            chain_idx_arr[:B], bead_start_arr[:B], n_moved_arr_in[:B],
            seg_type_arr[:B], anchor_a_arr[:B], anchor_b_arr[:B],
            angle_unif[:B],
            old_ca[:B], new_ca[:B], old_sg[:B], new_sg[:B],
            n_moved_arr_out[:B], move_type_arr[:B],
            N, box, inv_box, half_box,
            max_angle, l0, cap_inner, cap_tail,
        )

        delta_e = batch_delta_e_par(
            B, old_ca[:B], new_ca[:B], old_sg[:B], new_sg[:B],
            n_moved_arr_out[:B], chain_idx_arr[:B], bead_start_arr[:B],
            N, flat_pos,
            nc, inv_cs, half_box, neighbor_offsets,
            sorted_order, cell_starts, cell_counts,
            box, inv_box,
            r_rep_sq, r_max_sq, rep_e, contact_e,
            min_caca, min_casg, min_sgsg,
        )

        corr = correction_matrix_sparse(
            B, old_ca[:B], new_ca[:B], old_sg[:B], new_sg[:B],
            n_moved_arr_out[:B], chain_idx_arr[:B], bead_start_arr[:B], N,
            box, inv_box, r_max,
            r_rep_sq, r_max_sq, rep_e, contact_e,
            min_caca, min_casg, min_sgsg,
        )

        accepted = _fused_accept_and_correct(
            delta_e, corr, n_moved_arr_out[:B], kBT, accept_unif[:B])

        CHECKLIST_COUNTERS["ev_calls"] += int(B)
        for k_ in range(B):
            nm_k = int(n_moved_arr_out[k_])
            for j_ in range(nm_k):
                dx = new_ca[k_, j_, 0] - old_ca[k_, j_, 0]
                dy = new_ca[k_, j_, 1] - old_ca[k_, j_, 1]
                dz = new_ca[k_, j_, 2] - old_ca[k_, j_, 2]
                dx -= box * round(dx * inv_box)
                dy -= box * round(dy * inv_box)
                dz -= box * round(dz * inv_box)
                disp = math.sqrt(dx * dx + dy * dy + dz * dz)
                if disp > CHECKLIST_COUNTERS["max_disp"]:
                    CHECKLIST_COUNTERS["max_disp"] = disp
                if disp > l0 * 1.05:
                    CHECKLIST_COUNTERS["disp_violations"] += 1

        for i in range(B):
            nm = int(n_moved_arr_out[i])
            mtype_str = _MOVE_NAMES[int(move_type_arr[i])]
            stats.record(mtype_str, bool(accepted[i]))
            if nm == 0 or not accepted[i]:
                continue
            ci = int(chain_idx_arr[i])
            ms = int(bead_start_arr[i])
            for k in range(nm):
                state.ca[ci, ms + k, 0] = new_ca[i, k, 0]
                state.ca[ci, ms + k, 1] = new_ca[i, k, 1]
                state.ca[ci, ms + k, 2] = new_ca[i, k, 2]
                state.sg[ci, ms + k, 0] = new_sg[i, k, 0]
                state.sg[ci, ms + k, 1] = new_sg[i, k, 1]
                state.sg[ci, ms + k, 2] = new_sg[i, k, 2]
                ca_f = 2 * (ci * N + ms + k)
                sg_f = ca_f + 1
                flat_pos[ca_f, 0] = new_ca[i, k, 0]
                flat_pos[ca_f, 1] = new_ca[i, k, 1]
                flat_pos[ca_f, 2] = new_ca[i, k, 2]
                flat_pos[sg_f, 0] = new_sg[i, k, 0]
                flat_pos[sg_f, 1] = new_sg[i, k, 1]
                flat_pos[sg_f, 2] = new_sg[i, k, 2]
