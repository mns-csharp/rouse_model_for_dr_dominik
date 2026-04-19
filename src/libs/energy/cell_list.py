"""CellList — hybrid numpy-vectorized cell list for O(N) neighbor gather."""

import math
import numpy as np
import torch

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.number_space.number_space import NumberSpace


class CellList:
    def __init__(self, cfg: SimulationConfig, ns: NumberSpace):
        self.ns = ns
        self.r_max = cfg.r_max
        nc = max(3, int(math.floor(cfg.box_size / cfg.r_max)))
        self.n_cells = nc
        self.cell_size = cfg.box_size / nc
        self._half_box = ns.half_box
        self._inv_cs = 1.0 / self.cell_size
        self._nc = nc
        nc3 = nc * nc * nc
        self._nc3 = nc3
        self._neighbor_cells = [None] * nc3
        for lin in range(nc3):
            cx = lin // (nc * nc)
            cy = (lin // nc) % nc
            cz = lin % nc
            nbrs = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nx = (cx + dx) % nc
                        ny = (cy + dy) % nc
                        nz = (cz + dz) % nc
                        nbrs.append((nx * nc + ny) * nc + nz)
            self._neighbor_cells[lin] = nbrs
        self._sorted_order = np.empty(0, dtype=np.int64)
        self._cell_starts = np.zeros(nc3, dtype=np.int64)
        self._cell_counts = np.zeros(nc3, dtype=np.int64)

    def build(self, positions_cpu: torch.Tensor):
        nc = self._nc
        inv_cs = self._inv_cs
        half_box = self._half_box
        nc_m1 = nc - 1
        nc3 = self._nc3
        pos_np = positions_cpu.numpy()
        cx = np.clip(((pos_np[:, 0] + half_box) * inv_cs).astype(np.int64), 0, nc_m1)
        cy = np.clip(((pos_np[:, 1] + half_box) * inv_cs).astype(np.int64), 0, nc_m1)
        cz = np.clip(((pos_np[:, 2] + half_box) * inv_cs).astype(np.int64), 0, nc_m1)
        cell_idx = (cx * nc + cy) * nc + cz
        self._sorted_order = np.argsort(cell_idx, kind='mergesort')
        counts = np.bincount(cell_idx, minlength=nc3).astype(np.int64)
        starts = np.empty(nc3, dtype=np.int64)
        starts[0] = 0
        np.cumsum(counts[:-1], out=starts[1:])
        self._cell_counts = counts
        self._cell_starts = starts

    def gather_neighbors(self, old_positions, new_positions,
                         global_exclude_start: int,
                         global_exclude_end: int) -> list:
        nc = self._nc
        inv_cs = self._inv_cs
        half_box = self._half_box
        neighbor_cells = self._neighbor_cells
        nc_m1 = nc - 1
        sorted_order = self._sorted_order
        cell_starts = self._cell_starts
        cell_counts = self._cell_counts
        old_list = old_positions.tolist()
        new_list = new_positions.tolist()
        query_cells = set()
        for bead in old_list:
            cx = max(0, min(nc_m1, int((bead[0] + half_box) * inv_cs)))
            cy = max(0, min(nc_m1, int((bead[1] + half_box) * inv_cs)))
            cz = max(0, min(nc_m1, int((bead[2] + half_box) * inv_cs)))
            query_cells.add((cx * nc + cy) * nc + cz)
        for bead in new_list:
            cx = max(0, min(nc_m1, int((bead[0] + half_box) * inv_cs)))
            cy = max(0, min(nc_m1, int((bead[1] + half_box) * inv_cs)))
            cz = max(0, min(nc_m1, int((bead[2] + half_box) * inv_cs)))
            query_cells.add((cx * nc + cy) * nc + cz)
        nbr_cells = set()
        for qc in query_cells:
            nbr_cells.update(neighbor_cells[qc])
        slices = []
        for c in nbr_cells:
            cnt = cell_counts[c]
            if cnt > 0:
                s = cell_starts[c]
                slices.append(sorted_order[s:s + cnt])
        if not slices:
            return []
        all_atoms = np.concatenate(slices)
        gs = global_exclude_start
        ge = global_exclude_end
        mask = (all_atoms < gs) | (all_atoms >= ge)
        return all_atoms[mask].tolist()
