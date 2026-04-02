"""
GPU-resident cell list for neighbor queries.

Same algorithm as FastCellList (fast_energy.py) but all data lives on GPU
as PyTorch CUDA tensors. Build uses vectorized PyTorch ops (argsort, bincount).
Data is exposed to CuPy RawKernels via DLPack zero-copy interop.
"""

import math
import torch
import numpy as np


class GPUCellList:
    """Cell list stored entirely on GPU as PyTorch CUDA tensors."""

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

        # Pre-compute neighbor offsets [nc3, 27] on GPU
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
            assert idx == 27
        self.neighbor_offsets = torch.from_numpy(neighbor_offsets_np).to(device)

        # Populated by build()
        self.sorted_order = torch.empty(0, dtype=torch.int32, device=device)
        self.cell_starts = torch.zeros(nc3, dtype=torch.int32, device=device)
        self.cell_counts = torch.zeros(nc3, dtype=torch.int32, device=device)

    def build(self, positions_flat: torch.Tensor):
        """
        Build cell list from flat positions [total_beads, 3] on GPU.

        Fully vectorized using PyTorch ops on CUDA: cell assignment,
        argsort, bincount, cumsum. No Python per-bead loops.
        """
        nc = self._nc
        inv_cs = self._inv_cs
        half_box = self.half_box
        nc_m1 = nc - 1
        nc3 = self._nc3

        # Vectorized cell assignment on GPU
        pos = positions_flat.float()  # FP32 sufficient for cell assignment
        cx = ((pos[:, 0] + half_box) * inv_cs).int().clamp(0, nc_m1)
        cy = ((pos[:, 1] + half_box) * inv_cs).int().clamp(0, nc_m1)
        cz = ((pos[:, 2] + half_box) * inv_cs).int().clamp(0, nc_m1)
        cell_idx = (cx * nc + cy) * nc + cz

        # Sort atom indices by cell (stable sort preserves insertion order)
        self.sorted_order = torch.argsort(cell_idx, stable=True).int()

        # Cell offsets via bincount + cumsum
        counts = torch.bincount(cell_idx, minlength=nc3).int()
        starts = torch.zeros(nc3, dtype=torch.int32, device=self.device)
        if nc3 > 1:
            torch.cumsum(counts[:-1], dim=0, out=starts[1:])

        self.cell_counts = counts
        self.cell_starts = starts
