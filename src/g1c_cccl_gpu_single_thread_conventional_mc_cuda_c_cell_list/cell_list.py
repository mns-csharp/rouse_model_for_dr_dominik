"""GPU cell-list build via CuPy (matches g1c_ptgcl/cell_list.py semantics).

We use CuPy here instead of torch because the existing CUDA-C path passes
arrays to RawKernels via CuPy.  The cell-list outputs are int64 arrays:
  - `idx_sorted`  (NK,)        : bead indices sorted by cell_id
  - `cell_starts` (n_cells+1,) : prefix counts; bead range for cell c is
                                  idx_sorted[cell_starts[c] : cell_starts[c+1]]
"""

from __future__ import annotations

import math

import cupy as cp


def build_cell_index(positions_cp: cp.ndarray, box: float, r_cell: float):
    """positions_cp: (NK, 3) float32 wrapped or unwrapped. Returns
    (idx_sorted, cell_starts, n_cells_per_dim)."""
    NK = positions_cp.shape[0]
    n_cells_per_dim = max(1, int(box / r_cell))
    n_cells = n_cells_per_dim ** 3
    cell_w = box / n_cells_per_dim

    # Wrap into [0, box)
    pos = positions_cp - box * cp.floor(positions_cp / box)
    cell_xyz = cp.floor(pos / cell_w).astype(cp.int64)
    cell_xyz = cp.clip(cell_xyz, 0, n_cells_per_dim - 1)
    cell_id = (cell_xyz[:, 0]
               + cell_xyz[:, 1] * n_cells_per_dim
               + cell_xyz[:, 2] * n_cells_per_dim ** 2)

    sort_vals = cp.argsort(cell_id, kind="stable")
    sorted_cell_ids = cell_id[sort_vals]
    counts = cp.bincount(sorted_cell_ids, minlength=n_cells)
    cell_starts = cp.zeros(n_cells + 1, dtype=cp.int64)
    cell_starts[1:] = cp.cumsum(counts)
    return sort_vals.astype(cp.int64), cell_starts, n_cells_per_dim
