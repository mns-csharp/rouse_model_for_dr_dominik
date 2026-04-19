"""FastCellList — numpy-only cell list for neighbor queries (no PyTorch)."""

import math

import numpy as np

from rouse_model_python.src.libs.paths.fast.fast_kernels import FastKernels


class FastCellList:
    def __init__(self, box_size: float, r_max: float):
        nc = max(3, int(math.floor(box_size / r_max)))
        self.n_cells = nc
        self.cell_size = box_size / nc
        self.half_box = box_size / 2.0
        self._inv_cs = 1.0 / self.cell_size
        self._nc = nc
        self.box_size = box_size

        nc3 = nc * nc * nc
        self._nc3 = nc3

        self._neighbor_offsets = np.empty((nc3, 27), dtype=np.int32)
        self._neighbor_cells = [None] * nc3
        for lin in range(nc3):
            cx = lin // (nc * nc)
            cy = (lin // nc) % nc
            cz = lin % nc
            nbrs = []
            for ddx in range(-1, 2):
                for ddy in range(-1, 2):
                    for ddz in range(-1, 2):
                        nx = (cx + ddx) % nc
                        ny = (cy + ddy) % nc
                        nz = (cz + ddz) % nc
                        nbrs.append((nx * nc + ny) * nc + nz)
            self._neighbor_offsets[lin] = nbrs
            self._neighbor_cells[lin] = nbrs

        self._sorted_order = np.empty(0, dtype=np.int64)
        self._cell_starts = np.zeros(nc3, dtype=np.int64)
        self._cell_counts = np.zeros(nc3, dtype=np.int64)

    def build(self, pos_np: np.ndarray) -> None:
        nc = self._nc
        inv_cs = self._inv_cs
        half_box = self.half_box
        nc_m1 = nc - 1
        nc3 = self._nc3

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

    def gather_neighbors_jit(self, old_pos: np.ndarray, new_pos: np.ndarray,
                             global_exclude_start: int, global_exclude_end: int) -> np.ndarray:
        return FastKernels.gather_neighbors(
            old_pos, new_pos, self._nc, self._inv_cs, self.half_box,
            self._neighbor_offsets, self._sorted_order,
            self._cell_starts, self._cell_counts,
            global_exclude_start, global_exclude_end)
