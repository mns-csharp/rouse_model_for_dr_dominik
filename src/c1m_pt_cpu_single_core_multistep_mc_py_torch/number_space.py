"""NumberSpace — pure-numpy periodic coordinate system.

PBC wrap, MIC delta, cell-list grid mapping. Self-contained; no torch
dependency. The CPU multistep MC operates entirely on numpy arrays so
torch is not pulled into the import graph.
"""

from __future__ import annotations

import math
import numpy as np


class NumberSpace:
    """Cubic periodic box of side `box_size`. Wraps coords into
    [-half_box, +half_box). MIC distance via round-to-nearest-image."""

    def __init__(self, box_size: float, sigma: float = 3.8):
        if box_size <= 0:
            raise ValueError(f"box_size must be > 0; got {box_size}")
        self.box_size = float(box_size)
        self.half_box = self.box_size / 2.0
        self.sigma = float(sigma)
        self._inv_box = 1.0 / self.box_size
        self._inv_sigma = 1.0 / self.sigma

    # ── Core PBC ops ──────────────────────────────────────────────────
    def wrap(self, positions: np.ndarray) -> np.ndarray:
        return positions - self.box_size * np.floor(
            (positions + self.half_box) * self._inv_box)

    def wrap_scalar(self, v: float) -> float:
        return v - self.box_size * math.floor((v + self.half_box) * self._inv_box)

    def mic_delta(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        d = r2 - r1
        return d - self.box_size * np.round(d * self._inv_box)

    def mic_dist_sq(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        d = self.mic_delta(r1, r2)
        return (d * d).sum(axis=-1)

    # ── Cell-list grid ──────────────────────────────────────────────────
    def make_cell_grid(self, r_max: float):
        """Pre-compute the 27-neighbor table for a cubic cell grid sized
        so each cell has side >= r_max (so all interacting pairs sit
        within the central + 26 neighbors)."""
        nc = max(3, int(math.floor(self.box_size / r_max)))
        cell_size = self.box_size / nc
        nc3 = nc * nc * nc
        neighbor_offsets = np.empty((nc3, 27), dtype=np.int32)
        for lin in range(nc3):
            cx = lin // (nc * nc)
            cy = (lin // nc) % nc
            cz = lin % nc
            i = 0
            for ddx in range(-1, 2):
                for ddy in range(-1, 2):
                    for ddz in range(-1, 2):
                        nx = (cx + ddx) % nc
                        ny = (cy + ddy) % nc
                        nz = (cz + ddz) % nc
                        neighbor_offsets[lin, i] = (nx * nc + ny) * nc + nz
                        i += 1
        return nc, cell_size, neighbor_offsets

    def coord_to_cell_scalar(self, v: float, n_cells: int,
                             cell_size: float) -> int:
        c = int((v + self.half_box) / cell_size)
        return max(0, min(n_cells - 1, c))
