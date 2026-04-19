"""FastEnergyComputer — numpy/numba drop-in replacement for EnergyComputer."""

import numpy as np

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.paths.fast.fast_cell_list import FastCellList
from rouse_model_python.src.libs.paths.fast.fast_kernels import FastKernels


class FastEnergyComputer:
    def __init__(self, cfg: SimulationConfig):
        self.box = cfg.box_size
        self.inv_box = 1.0 / cfg.box_size
        self.r_rep_sq = cfg.r_rep_sq
        self.r_max_sq = cfg.r_max_sq
        self.rep_e = cfg.repulsive_energy
        self.contact_e = cfg.contact_energy
        self.kBT = cfg.kBT

        self.cell_list = FastCellList(cfg.box_size, cfg.r_max)
        self._pos_np = None

        self._jit_warmed = False

    def _warmup_jit(self) -> None:
        if self._jit_warmed:
            return
        dummy = np.zeros((2, 3), dtype=np.float64)
        FastKernels.count_overlaps(dummy, dummy, self.box, self.inv_box, self.r_rep_sq)
        FastKernels.delta_energy_fast(dummy, dummy, dummy, self.box, self.inv_box,
                                      self.r_rep_sq, self.rep_e)
        FastKernels.compute_segment_pair_energy(dummy, dummy, self.box, self.inv_box,
                                                self.r_rep_sq, self.rep_e)
        self._jit_warmed = True

    def rebuild_cell_list(self, pos_np: np.ndarray) -> None:
        self._pos_np = pos_np
        self.cell_list.build(pos_np)
        if not self._jit_warmed:
            self._warmup_jit()

    def compute_delta_energy(self, chain_idx: int, bead_start: int, n_moved: int,
                             old_pos_np: np.ndarray, new_pos_np: np.ndarray,
                             N: int) -> float:
        global_start = chain_idx * N + bead_start
        global_end = global_start + n_moved
        cl = self.cell_list

        if self.contact_e != 0.0:
            return FastKernels.delta_energy_direct_3zone(
                old_pos_np, new_pos_np, self._pos_np,
                cl._nc, cl._inv_cs, cl.half_box,
                cl._neighbor_offsets, cl._sorted_order,
                cl._cell_starts, cl._cell_counts,
                global_start, global_end,
                self.box, self.inv_box, self.r_rep_sq, self.r_max_sq,
                self.rep_e, self.contact_e)
        return FastKernels.delta_energy_direct(
            old_pos_np, new_pos_np, self._pos_np,
            cl._nc, cl._inv_cs, cl.half_box,
            cl._neighbor_offsets, cl._sorted_order,
            cl._cell_starts, cl._cell_counts,
            global_start, global_end,
            self.box, self.inv_box, self.r_rep_sq, self.rep_e)

    def compute_batch_delta_energy(self, proposals, N: int, max_moved: int):
        B = len(proposals)
        old_flat = np.zeros((B, max_moved, 3), dtype=np.float64)
        new_flat = np.zeros((B, max_moved, 3), dtype=np.float64)
        nm_arr = np.zeros(B, dtype=np.int64)
        gs_arr = np.zeros(B, dtype=np.int64)
        ge_arr = np.zeros(B, dtype=np.int64)

        for i, p in enumerate(proposals):
            nm = p.n_moved
            if nm > 0:
                old_flat[i, :nm, :] = p.old_pos
                new_flat[i, :nm, :] = p.new_pos
                nm_arr[i] = nm
                gs_arr[i] = p.chain_idx * N + p.bead_start
                ge_arr[i] = gs_arr[i] + nm

        cl = self.cell_list
        results = FastKernels.batch_delta_energy_direct(
            B, old_flat, new_flat, nm_arr,
            self._pos_np, gs_arr, ge_arr,
            cl._nc, cl._inv_cs, cl.half_box,
            cl._neighbor_offsets, cl._sorted_order,
            cl._cell_starts, cl._cell_counts,
            self.box, self.inv_box, self.r_rep_sq, self.rep_e,
            max_moved)

        return results.tolist()

    def compute_segment_pair_energy(self, pos_a_np: np.ndarray, pos_b_np: np.ndarray) -> float:
        return FastKernels.compute_segment_pair_energy(
            pos_a_np, pos_b_np, self.box, self.inv_box,
            self.r_rep_sq, self.rep_e)
