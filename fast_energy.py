"""
Fast energy computation using numpy + numba JIT.

Replaces the PyTorch-based delta-E path in energy.py for the sequential
(CPU) mode. Key optimizations:
  1. Numba-JIT'd pairwise overlap counting (eliminates Python/PyTorch overhead)
  2. Numpy cell-list gather (already fast, unchanged)
  3. Skips intra-segment energy (all MC moves are rigid-body rotations)
  4. Pre-stores positions as contiguous numpy array
"""

import numpy as np
import math
from numba import njit, prange, types
from numba.typed import List as NumbaList

# ---------------------------------------------------------------------------
# Numba-JIT'd core kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _count_overlaps(moved_pos, nbr_pos, box, inv_box, r_rep_sq):
    """Count overlap pairs between moved beads and neighbor beads.

    Args:
        moved_pos: [M, 3] float64 array
        nbr_pos: [K, 3] float64 array
        box, inv_box: PBC box parameters
        r_rep_sq: squared repulsive cutoff

    Returns:
        int count of pairs with r² < r_rep_sq
    """
    M = moved_pos.shape[0]
    K = nbr_pos.shape[0]
    count = 0
    for i in range(M):
        mx = moved_pos[i, 0]
        my = moved_pos[i, 1]
        mz = moved_pos[i, 2]
        for j in range(K):
            dx = nbr_pos[j, 0] - mx
            dy = nbr_pos[j, 1] - my
            dz = nbr_pos[j, 2] - mz
            dx -= box * round(dx * inv_box)
            dy -= box * round(dy * inv_box)
            dz -= box * round(dz * inv_box)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < r_rep_sq:
                count += 1
    return count


@njit(cache=True)
def _delta_energy_fast(old_pos, new_pos, nbr_pos, box, inv_box, r_rep_sq, rep_e):
    """Compute delta-E for a single move proposal using numba.

    Skips intra-segment energy (rigid-body moves preserve it).

    Returns:
        float delta_e
    """
    e_old = _count_overlaps(old_pos, nbr_pos, box, inv_box, r_rep_sq)
    e_new = _count_overlaps(new_pos, nbr_pos, box, inv_box, r_rep_sq)
    return (e_new - e_old) * rep_e


@njit(cache=True)
def _compute_segment_pair_energy_fast(pos_a, pos_b, box, inv_box, r_rep_sq, rep_e):
    """Compute pairwise excluded-volume energy between two groups of beads."""
    return _count_overlaps(pos_a, pos_b, box, inv_box, r_rep_sq) * rep_e


@njit(cache=True)
def _gather_neighbors_jit(old_pos, new_pos, nc, inv_cs, half_box,
                           neighbor_offsets, sorted_order, cell_starts,
                           cell_counts, global_exclude_start, global_exclude_end):
    """
    Numba-JIT'd neighbor gathering. Replaces Python set operations + numpy
    concatenation with a single compiled function.

    neighbor_offsets: [nc3, 27] int32 array of precomputed neighbor cell indices

    Returns:
        numpy int64 array of neighbor atom indices (excluding the moved range)
    """
    nc_m1 = nc - 1
    M_old = old_pos.shape[0]
    M_new = new_pos.shape[0]
    nc3 = nc * nc * nc

    # Bitmap for visited neighbor cells (up to ~1M cells fits in stack)
    visited = np.zeros(nc3, dtype=np.bool_)

    # Find cells for old + new beads, expand to neighbors
    for k in range(M_old):
        cx = max(0, min(nc_m1, int((old_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((old_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((old_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            visited[neighbor_offsets[cell, ni]] = True

    for k in range(M_new):
        cx = max(0, min(nc_m1, int((new_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((new_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((new_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            visited[neighbor_offsets[cell, ni]] = True

    # Count total atoms in visited cells (for pre-allocation)
    total = 0
    for c in range(nc3):
        if visited[c]:
            total += cell_counts[c]

    # Gather atom indices
    result = np.empty(total, dtype=np.int64)
    idx = 0
    gs = global_exclude_start
    ge = global_exclude_end
    for c in range(nc3):
        if not visited[c]:
            continue
        cnt = cell_counts[c]
        if cnt == 0:
            continue
        s = cell_starts[c]
        for i in range(cnt):
            atom = sorted_order[s + i]
            if atom < gs or atom >= ge:
                result[idx] = atom
                idx += 1

    return result[:idx]


@njit(cache=True)
def _delta_energy_direct(old_pos, new_pos, all_pos, nc, inv_cs, half_box,
                          neighbor_offsets, sorted_order, cell_starts,
                          cell_counts, global_exclude_start, global_exclude_end,
                          box, inv_box, r_rep_sq, rep_e):
    """
    Combined gather + overlap counting in a single JIT'd function.
    Eliminates the intermediate neighbor array allocation.
    """
    nc_m1 = nc - 1
    M_old = old_pos.shape[0]
    M_new = new_pos.shape[0]
    nc3 = nc * nc * nc
    gs = global_exclude_start
    ge = global_exclude_end

    # Bitmap for visited neighbor cells
    visited = np.zeros(nc3, dtype=np.bool_)

    for k in range(M_old):
        cx = max(0, min(nc_m1, int((old_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((old_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((old_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            visited[neighbor_offsets[cell, ni]] = True
    for k in range(M_new):
        cx = max(0, min(nc_m1, int((new_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((new_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((new_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            visited[neighbor_offsets[cell, ni]] = True

    # Count overlaps directly (no intermediate array)
    e_old = 0
    e_new = 0
    for c in range(nc3):
        if not visited[c]:
            continue
        cnt = cell_counts[c]
        if cnt == 0:
            continue
        s = cell_starts[c]
        for i in range(cnt):
            atom = sorted_order[s + i]
            if atom >= gs and atom < ge:
                continue
            nx = all_pos[atom, 0]
            ny = all_pos[atom, 1]
            nz = all_pos[atom, 2]
            for m in range(M_old):
                dx = nx - old_pos[m, 0]
                dy = ny - old_pos[m, 1]
                dz = nz - old_pos[m, 2]
                dx -= box * round(dx * inv_box)
                dy -= box * round(dy * inv_box)
                dz -= box * round(dz * inv_box)
                r2 = dx*dx + dy*dy + dz*dz
                if r2 < r_rep_sq:
                    e_old += 1
            for m in range(M_new):
                dx = nx - new_pos[m, 0]
                dy = ny - new_pos[m, 1]
                dz = nz - new_pos[m, 2]
                dx -= box * round(dx * inv_box)
                dy -= box * round(dy * inv_box)
                dz -= box * round(dz * inv_box)
                r2 = dx*dx + dy*dy + dz*dz
                if r2 < r_rep_sq:
                    e_new += 1

    return (e_new - e_old) * rep_e


# ---------------------------------------------------------------------------
# Fast cell-list (numpy-only, same algorithm as energy.py CellList)
# ---------------------------------------------------------------------------

class FastCellList:
    """Numpy-only cell list for neighbor queries. No PyTorch dependency."""

    def __init__(self, box_size, r_max):
        nc = max(3, int(math.floor(box_size / r_max)))
        self.n_cells = nc
        self.cell_size = box_size / nc
        self.half_box = box_size / 2.0
        self._inv_cs = 1.0 / self.cell_size
        self._nc = nc
        self.box_size = box_size

        nc3 = nc * nc * nc
        self._nc3 = nc3

        # Pre-compute neighbor offsets as contiguous [nc3, 27] numpy array
        # for fast numba access
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

    def build(self, pos_np):
        """Build from numpy [total_beads, 3] array."""
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

    def gather_neighbors_jit(self, old_pos, new_pos,
                              global_exclude_start, global_exclude_end):
        """JIT'd neighbor gathering."""
        return _gather_neighbors_jit(
            old_pos, new_pos, self._nc, self._inv_cs, self.half_box,
            self._neighbor_offsets, self._sorted_order,
            self._cell_starts, self._cell_counts,
            global_exclude_start, global_exclude_end)


# ---------------------------------------------------------------------------
# FastEnergyComputer: drop-in replacement for sequential CPU mode
# ---------------------------------------------------------------------------

class FastEnergyComputer:
    """
    Fast energy computer using numpy + numba for sequential CPU mode.

    Drop-in replacement for EnergyComputer's delta-E path.
    All positions stored as numpy arrays. No PyTorch in the hot path.
    """

    def __init__(self, cfg):
        self.box = cfg.box_size
        self.inv_box = 1.0 / cfg.box_size
        self.r_rep_sq = cfg.r_rep_sq
        self.r_max_sq = cfg.r_max_sq
        self.rep_e = cfg.repulsive_energy
        self.kBT = cfg.kBT

        self.cell_list = FastCellList(cfg.box_size, cfg.r_max)
        self._pos_np = None  # [total_beads, 3] numpy f64

        # Warm up numba JIT on first call
        self._jit_warmed = False

    def _warmup_jit(self):
        """Force-compile numba kernels with small dummy arrays."""
        if self._jit_warmed:
            return
        dummy = np.zeros((2, 3), dtype=np.float64)
        _count_overlaps(dummy, dummy, self.box, self.inv_box, self.r_rep_sq)
        _delta_energy_fast(dummy, dummy, dummy, self.box, self.inv_box,
                           self.r_rep_sq, self.rep_e)
        _compute_segment_pair_energy_fast(dummy, dummy, self.box, self.inv_box,
                                          self.r_rep_sq, self.rep_e)
        self._jit_warmed = True

    def rebuild_cell_list(self, pos_np):
        """Rebuild from numpy [total_beads, 3] positions."""
        self._pos_np = pos_np
        self.cell_list.build(pos_np)
        if not self._jit_warmed:
            self._warmup_jit()

    def compute_delta_energy(self, chain_idx, bead_start, n_moved,
                              old_pos_np, new_pos_np, N):
        """
        Compute ΔE for a proposed move. Uses combined gather+overlap JIT kernel.

        Args:
            chain_idx: int
            bead_start: int
            n_moved: int
            old_pos_np: [M, 3] numpy array (contiguous)
            new_pos_np: [M, 3] numpy array (contiguous)
            N: beads per chain

        Returns:
            float delta_e
        """
        global_start = chain_idx * N + bead_start
        global_end = global_start + n_moved
        cl = self.cell_list

        return _delta_energy_direct(
            old_pos_np, new_pos_np, self._pos_np,
            cl._nc, cl._inv_cs, cl.half_box,
            cl._neighbor_offsets, cl._sorted_order,
            cl._cell_starts, cl._cell_counts,
            global_start, global_end,
            self.box, self.inv_box, self.r_rep_sq, self.rep_e)

    def compute_segment_pair_energy(self, pos_a_np, pos_b_np):
        """Compute pairwise energy between two groups of beads (for rank-1)."""
        return _compute_segment_pair_energy_fast(
            pos_a_np, pos_b_np, self.box, self.inv_box,
            self.r_rep_sq, self.rep_e)
