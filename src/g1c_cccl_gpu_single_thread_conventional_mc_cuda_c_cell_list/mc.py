"""Cell-list MC sweep — per-batch CUDA-C kernel dispatch.

Replaces the mega-kernel pattern of the all-pairs g1c_cc with a per-round
batch loop:

  for s in shuffled(segs_per_chain):
    rows = chain_order * spc + s            # numpy int64
    build CA + SG cell lists                # CuPy
    propose_batch_kernel  (B = n_chains)
    delta_e_cell_list_segment_kernel        # NEW; takes cell-list args
    accept_segment_kernel (B = n_chains)

The cell list bounds the per-bead pair-energy work from O(NK) to
O(<n_per_cell> * 27), unlocking large-K simulations. Built once per round
(same approach as g1c_ptgcl).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .chain import ChainState
from .config import SimConfig
from .proposer import MTYPE_HINGE, MTYPE_N_TAIL, MTYPE_C_TAIL
from . import cuda_kernels as ck
from . import cell_list as _cl


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
    """Run one full sweep on the GPU via per-batch CUDA-C kernel dispatch.

    Returns: cumulative kernel ms across all per-round launches in this sweep.
    """
    import cupy as cp

    seg_info = state.segments
    N = cfg.N
    n_chains = cfg.n_chains
    box = cfg.box_size
    kBT = cfg.kBT
    M = scratch.M
    r_cell = max(float(cfg.r_max), float(cfg.sigma))

    spc = seg_info.segs_per_chain
    total_segs = seg_info.total_segments

    scratch.fill_rand(rng)
    scratch.reset_counters()

    chain_order = np.arange(n_chains, dtype=np.int64)
    rng.shuffle(chain_order)
    round_order = np.arange(spc, dtype=np.int64)
    rng.shuffle(round_order)

    inv_box = 1.0 / box
    half_box = 0.5 * box
    inv_kBT = 1.0 / float(kBT)

    cp_ca = scratch.cp_state_ca(state.ca)
    cp_sg = scratch.cp_state_sg(state.sg)
    cp_old_ca = scratch.cp("old_ca")
    cp_new_ca = scratch.cp("new_ca")
    cp_old_sg = scratch.cp("old_sg")
    cp_new_sg = scratch.cp("new_sg")
    cp_n_moved_out = scratch.cp("n_moved_out")
    cp_move_type = scratch.cp("move_type")
    cp_delta_e = scratch.cp("delta_e")
    cp_att = scratch.cp("attempted_counts")
    cp_acc = scratch.cp("accepted_counts")
    cp_col_chain = scratch.cp("col_chain")
    cp_col_bead = scratch.cp("col_bead")
    cp_col_nmov = scratch.cp("col_nmov")
    cp_col_seg = scratch.cp("col_seg")
    cp_col_aa = scratch.cp("col_aa")
    cp_col_ab = scratch.cp("col_ab")
    cp_rand_buf = scratch.cp("rand_buf")
    cp_rand_accept = scratch.cp("rand_accept")

    propose_kernel = ck.get_propose_kernel()
    delta_e_cl_kernel = ck.get_delta_e_cell_list_kernel()
    accept_kernel = ck.get_accept_kernel()

    torch_stream = torch.cuda.current_stream(state.ca.device)
    cp_stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)

    B = int(n_chains)
    block_propose = (32, 1, 1)
    block_dE = (256, 1, 1)
    block_accept = (32, 1, 1)
    grid = (B, 1, 1)
    smem_propose = (M * 3 + 8) * 4
    smem_dE = 256 * 4

    chain_order_cp = cp.asarray(chain_order)
    cell_w = box / max(1, int(box / r_cell))

    total_kernel_ms = 0.0
    evt_start = cp.cuda.Event()
    evt_end = cp.cuda.Event()

    for s in round_order:
        # Build cell lists for this round (from current state).
        idx_ca, starts_ca, n_cells_per_dim = _cl.build_cell_index(
            cp_ca, box, r_cell)
        idx_sg, starts_sg, _ = _cl.build_cell_index(cp_sg, box, r_cell)

        # gs_round[i] = chain_order[i] * spc + s — global segment IDs for round.
        gs_round = chain_order_cp * spc + int(s)
        gs_round_i64 = gs_round.astype(cp.int64)

        # Gather per-segment metadata via fancy indexing.
        cp_ch = cp_col_chain[gs_round_i64].astype(cp.int64)
        cp_bs = cp_col_bead[gs_round_i64].astype(cp.int64)
        cp_nm = cp_col_nmov[gs_round_i64].astype(cp.int64)
        cp_st = cp_col_seg[gs_round_i64].astype(cp.int64)
        cp_aa = cp_col_aa[gs_round_i64].astype(cp.int64)
        cp_ab = cp_col_ab[gs_round_i64].astype(cp.int64)
        cp_ra = cp_rand_buf[gs_round_i64].astype(cp.float32)
        cp_rc = cp_rand_accept[gs_round_i64].astype(cp.float32)

        with cp_stream:
            evt_start.record()
            # Phase 1: propose
            propose_kernel(grid, block_propose, (
                cp_ca, cp_sg,
                cp_old_ca, cp_new_ca, cp_old_sg, cp_new_sg,
                cp_ch, cp_bs, cp_nm, cp_st, cp_aa, cp_ab,
                cp_ra,
                cp_n_moved_out, cp_move_type,
                np.int32(N), np.int32(M), np.int32(B),
                np.float32(box), np.float32(inv_box), np.float32(half_box),
                np.float32(cfg.max_angle_hinge), np.float32(0.0),
            ), shared_mem=smem_propose)

            # Phase 2: delta_e via cell list
            delta_e_cl_kernel(grid, block_dE, (
                cp_ca, cp_sg,
                cp_old_ca, cp_new_ca, cp_old_sg, cp_new_sg,
                cp_n_moved_out, cp_ch, cp_bs,
                idx_ca, starts_ca, idx_sg, starts_sg,
                np.int32(n_cells_per_dim), np.float32(cell_w),
                cp_delta_e,
                np.int32(M), np.int32(B),
                np.int32(n_chains), np.int32(N),
                np.float32(box), np.float32(inv_box),
                np.float32(cfg.r_rep_sq), np.float32(cfg.r_max_sq),
                np.float32(cfg.repulsive_energy), np.float32(cfg.contact_energy),
                np.int32(cfg.min_seq_caca), np.int32(cfg.min_seq_casg),
                np.int32(cfg.min_seq_sgsg),
            ), shared_mem=smem_dE)

            # Phase 3: accept + writeback
            accept_kernel(grid, block_accept, (
                cp_ca, cp_sg, cp_new_ca, cp_new_sg,
                cp_delta_e, cp_n_moved_out, cp_move_type,
                cp_ch, cp_bs, cp_rc,
                cp_att, cp_acc,
                np.int32(M), np.int32(B), np.int32(N),
                np.float32(inv_kBT),
            ))
            evt_end.record()
        evt_end.synchronize()
        total_kernel_ms += float(cp.cuda.get_elapsed_time(evt_start, evt_end))

    attempted_h = scratch.attempted_counts.cpu().numpy()
    accepted_h = scratch.accepted_counts.cpu().numpy()
    stats.add_batch(attempted_h, accepted_h)
    CHECKLIST_COUNTERS["ev_calls"] += int(attempted_h.sum())
    return total_kernel_ms
