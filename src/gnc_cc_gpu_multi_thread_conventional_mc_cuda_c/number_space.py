"""NumberSpace — torch.cuda PBC/MIC operations."""

from __future__ import annotations

import math

import numpy as np
import torch


class NumberSpace:
    def __init__(self, box_size: float, sigma: float, device: torch.device,
                 dtype: torch.dtype = torch.float32):
        self.box_size = float(box_size)
        self.half_box = self.box_size / 2.0
        self.sigma = float(sigma)
        self.device = device
        self.dtype = dtype
        self._inv_box = 1.0 / self.box_size

    def wrap(self, p: torch.Tensor) -> torch.Tensor:
        return p - self.box_size * torch.floor((p + self.half_box) * self._inv_box)

    def mic_delta(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        d = r2 - r1
        return d - self.box_size * torch.round(d * self._inv_box)

    def mic_dist_sq(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        d = self.mic_delta(r1, r2)
        return (d * d).sum(dim=-1)

    def make_cell_grid(self, r_max: float):
        nc = max(3, int(math.floor(self.box_size / r_max)))
        cell_size = self.box_size / nc
        nc3 = nc * nc * nc
        offs = np.empty((nc3, 27), dtype=np.int32)
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
                        offs[lin, i] = (nx * nc + ny) * nc + nz
                        i += 1
        return nc, cell_size, offs
