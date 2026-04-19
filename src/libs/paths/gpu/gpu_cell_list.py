"""GPUCellList — GPU-resident cell list using PyTorch CUDA tensors."""

import math

import numpy as np
import torch


class GPUCellList:
    def __init__(self, box_size: float, r_max: float, device: torch.device):
        nc = max(3, int(math.floor(box_size / r_max)))
        self.n_cells = nc
        self.cell_size = box_size / nc
        self.half_box = box_size / 2.0
        self._inv_cs = 1.0 / self.cell_size
        self._nc = nc
        self.box_size = box_size
        self.device = device

        nc3 = nc * nc * nc
        self._nc3 = nc3

        neighbor_offsets_np = np.empty((nc3, 27), dtype=np.int32)
        for lin in range(nc3):
            cx = lin // (nc * nc)
            cy = (lin // nc) % nc
            cz = lin % nc
            idx = 0
            for ddx in range(-1, 2):
                for ddy in range(-1, 2):
                    for ddz in range(-1, 2):
                        nx = (cx + ddx) % nc
                        ny = (cy + ddy) % nc
                        nz = (cz + ddz) % nc
                        neighbor_offsets_np[lin, idx] = (nx * nc + ny) * nc + nz
                        idx += 1
        self.neighbor_offsets = torch.from_numpy(neighbor_offsets_np).to(device)

        self.sorted_order = torch.empty(0, dtype=torch.int32, device=device)
        self.cell_starts = torch.zeros(nc3, dtype=torch.int32, device=device)
        self.cell_counts = torch.zeros(nc3, dtype=torch.int32, device=device)

    def build(self, positions_flat: torch.Tensor) -> None:
        nc = self._nc
        inv_cs = self._inv_cs
        half_box = self.half_box
        nc_m1 = nc - 1
        nc3 = self._nc3

        pos = positions_flat.float()
        cx = ((pos[:, 0] + half_box) * inv_cs).int().clamp(0, nc_m1)
        cy = ((pos[:, 1] + half_box) * inv_cs).int().clamp(0, nc_m1)
        cz = ((pos[:, 2] + half_box) * inv_cs).int().clamp(0, nc_m1)
        cell_idx = (cx * nc + cy) * nc + cz

        self.sorted_order = torch.argsort(cell_idx, stable=True).int()

        counts = torch.bincount(cell_idx, minlength=nc3).int()
        starts = torch.zeros(nc3, dtype=torch.int32, device=self.device)
        if nc3 > 1:
            torch.cumsum(counts[:-1], dim=0, out=starts[1:])

        self.cell_counts = counts
        self.cell_starts = starts
