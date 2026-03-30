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
def _three_zone_energy(moved_pos, nbr_pos, box, inv_box,
                       r_rep_sq, r_max_sq, rep_e, contact_e):
    """Compute 3-zone energy: repulsive (r<r_rep) + contact (r_rep<=r<r_max).

    Returns:
        float total energy contribution
    """
    M = moved_pos.shape[0]
    K = nbr_pos.shape[0]
    energy = 0.0
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
                energy += rep_e
            elif r2 < r_max_sq:
                energy += contact_e
    return energy


@njit(cache=True)
def _delta_energy_fast(old_pos, new_pos, nbr_pos, box, inv_box, r_rep_sq, rep_e):
    """Compute delta-E for a single move proposal using numba (repulsive only).

    Skips intra-segment energy (rigid-body moves preserve it).

    Returns:
        float delta_e
    """
    e_old = _count_overlaps(old_pos, nbr_pos, box, inv_box, r_rep_sq)
    e_new = _count_overlaps(new_pos, nbr_pos, box, inv_box, r_rep_sq)
    return (e_new - e_old) * rep_e


@njit(cache=True)
def _delta_energy_3zone(old_pos, new_pos, nbr_pos, box, inv_box,
                        r_rep_sq, r_max_sq, rep_e, contact_e):
    """Compute delta-E using 3-zone kernel (repulsive + contact)."""
    e_old = _three_zone_energy(old_pos, nbr_pos, box, inv_box,
                               r_rep_sq, r_max_sq, rep_e, contact_e)
    e_new = _three_zone_energy(new_pos, nbr_pos, box, inv_box,
                               r_rep_sq, r_max_sq, rep_e, contact_e)
    return e_new - e_old


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
    Uses visited-cell list instead of full bitmap scan for O(K) instead
    of O(nc³) iteration — critical for large boxes with many empty cells.
    """
    nc_m1 = nc - 1
    M_old = old_pos.shape[0]
    M_new = new_pos.shape[0]
    nc3 = nc * nc * nc
    gs = global_exclude_start
    ge = global_exclude_end

    # Bitmap + visited-cell list: O(K) iteration instead of O(nc³) scan
    visited = np.zeros(nc3, dtype=np.bool_)
    visited_list = np.empty((M_old + M_new) * 27, dtype=np.int64)
    n_visited = 0

    for k in range(M_old):
        cx = max(0, min(nc_m1, int((old_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((old_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((old_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            c = neighbor_offsets[cell, ni]
            if not visited[c]:
                visited[c] = True
                visited_list[n_visited] = c
                n_visited += 1
    for k in range(M_new):
        cx = max(0, min(nc_m1, int((new_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((new_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((new_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            c = neighbor_offsets[cell, ni]
            if not visited[c]:
                visited[c] = True
                visited_list[n_visited] = c
                n_visited += 1

    # Count overlaps — iterate only visited cells
    e_old = 0
    e_new = 0
    for vi in range(n_visited):
        c = visited_list[vi]
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


@njit(parallel=True, cache=True)
def _batch_delta_energy_direct(
        n_proposals,
        old_pos_flat, new_pos_flat, n_moved_arr,
        all_pos, global_starts, global_ends,
        nc, inv_cs, half_box,
        neighbor_offsets, sorted_order, cell_starts, cell_counts,
        box, inv_box, r_rep_sq, rep_e,
        max_moved):
    """
    Compute delta-E for B proposals in parallel using numba prange.

    Each proposal is processed independently on a separate thread.
    All proposals must be from different chains (guaranteed by round-robin).

    Args:
        n_proposals: number of valid proposals
        old_pos_flat: [B, max_moved, 3] old positions (padded)
        new_pos_flat: [B, max_moved, 3] new positions (padded)
        n_moved_arr: [B] number of moved beads per proposal
        all_pos: [total_beads, 3] all positions
        global_starts: [B] global bead index start
        global_ends: [B] global bead index end
        nc, inv_cs, half_box: cell-list params
        neighbor_offsets: [nc3, 27] neighbor cell offsets
        sorted_order, cell_starts, cell_counts: cell-list data
        box, inv_box, r_rep_sq, rep_e: energy params
        max_moved: max beads moved in any proposal

    Returns:
        [B] array of delta-E values
    """
    B = n_proposals
    nc_m1 = nc - 1
    nc3 = nc * nc * nc
    results = np.zeros(B, dtype=np.float64)

    for pi in prange(B):
        nm = n_moved_arr[pi]
        if nm == 0:
            continue
        gs = global_starts[pi]
        ge = global_ends[pi]

        old_p = old_pos_flat[pi, :nm, :]
        new_p = new_pos_flat[pi, :nm, :]

        # Bitmap for visited neighbor cells
        visited = np.zeros(nc3, dtype=np.bool_)
        for k in range(nm):
            cx = max(0, min(nc_m1, int((old_p[k, 0] + half_box) * inv_cs)))
            cy = max(0, min(nc_m1, int((old_p[k, 1] + half_box) * inv_cs)))
            cz = max(0, min(nc_m1, int((old_p[k, 2] + half_box) * inv_cs)))
            cell = (cx * nc + cy) * nc + cz
            for ni in range(27):
                visited[neighbor_offsets[cell, ni]] = True
            cx = max(0, min(nc_m1, int((new_p[k, 0] + half_box) * inv_cs)))
            cy = max(0, min(nc_m1, int((new_p[k, 1] + half_box) * inv_cs)))
            cz = max(0, min(nc_m1, int((new_p[k, 2] + half_box) * inv_cs)))
            cell = (cx * nc + cy) * nc + cz
            for ni in range(27):
                visited[neighbor_offsets[cell, ni]] = True

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
                for m in range(nm):
                    dx = nx - old_p[m, 0]
                    dy = ny - old_p[m, 1]
                    dz = nz - old_p[m, 2]
                    dx -= box * round(dx * inv_box)
                    dy -= box * round(dy * inv_box)
                    dz -= box * round(dz * inv_box)
                    r2 = dx*dx + dy*dy + dz*dz
                    if r2 < r_rep_sq:
                        e_old += 1
                for m in range(nm):
                    dx = nx - new_p[m, 0]
                    dy = ny - new_p[m, 1]
                    dz = nz - new_p[m, 2]
                    dx -= box * round(dx * inv_box)
                    dy -= box * round(dy * inv_box)
                    dz -= box * round(dz * inv_box)
                    r2 = dx*dx + dy*dy + dz*dz
                    if r2 < r_rep_sq:
                        e_new += 1

        results[pi] = (e_new - e_old) * rep_e

    return results


@njit(cache=True)
def _delta_energy_direct_3zone(old_pos, new_pos, all_pos, nc, inv_cs, half_box,
                                neighbor_offsets, sorted_order, cell_starts,
                                cell_counts, global_exclude_start, global_exclude_end,
                                box, inv_box, r_rep_sq, r_max_sq, rep_e, contact_e):
    """
    Combined gather + 3-zone energy in a single JIT'd function.
    Adds contact zone (r_rep <= r < r_max) energy to the repulsive zone.
    """
    nc_m1 = nc - 1
    M_old = old_pos.shape[0]
    M_new = new_pos.shape[0]
    nc3 = nc * nc * nc
    gs = global_exclude_start
    ge = global_exclude_end

    visited = np.zeros(nc3, dtype=np.bool_)
    visited_list = np.empty((M_old + M_new) * 27, dtype=np.int64)
    n_visited = 0

    for k in range(M_old):
        cx = max(0, min(nc_m1, int((old_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((old_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((old_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            c = neighbor_offsets[cell, ni]
            if not visited[c]:
                visited[c] = True
                visited_list[n_visited] = c
                n_visited += 1
    for k in range(M_new):
        cx = max(0, min(nc_m1, int((new_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((new_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((new_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            c = neighbor_offsets[cell, ni]
            if not visited[c]:
                visited[c] = True
                visited_list[n_visited] = c
                n_visited += 1

    e_old = 0.0
    e_new = 0.0
    for vi in range(n_visited):
        c = visited_list[vi]
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
                    e_old += rep_e
                elif r2 < r_max_sq:
                    e_old += contact_e
            for m in range(M_new):
                dx = nx - new_pos[m, 0]
                dy = ny - new_pos[m, 1]
                dz = nz - new_pos[m, 2]
                dx -= box * round(dx * inv_box)
                dy -= box * round(dy * inv_box)
                dz -= box * round(dz * inv_box)
                r2 = dx*dx + dy*dy + dz*dz
                if r2 < r_rep_sq:
                    e_new += rep_e
                elif r2 < r_max_sq:
                    e_new += contact_e

    return e_new - e_old


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
        self.contact_e = cfg.contact_energy
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
        Automatically selects 3-zone kernel when contact_energy != 0.

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

        if self.contact_e != 0.0:
            return _delta_energy_direct_3zone(
                old_pos_np, new_pos_np, self._pos_np,
                cl._nc, cl._inv_cs, cl.half_box,
                cl._neighbor_offsets, cl._sorted_order,
                cl._cell_starts, cl._cell_counts,
                global_start, global_end,
                self.box, self.inv_box, self.r_rep_sq, self.r_max_sq,
                self.rep_e, self.contact_e)
        return _delta_energy_direct(
            old_pos_np, new_pos_np, self._pos_np,
            cl._nc, cl._inv_cs, cl.half_box,
            cl._neighbor_offsets, cl._sorted_order,
            cl._cell_starts, cl._cell_counts,
            global_start, global_end,
            self.box, self.inv_box, self.r_rep_sq, self.rep_e)

    def compute_batch_delta_energy(self, proposals, N, max_moved):
        """
        Compute delta-E for a batch of proposals in parallel using numba prange.

        Args:
            proposals: list of FastProposal objects
            N: beads per chain
            max_moved: max n_moved across proposals

        Returns:
            list of float delta-E values
        """
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
        results = _batch_delta_energy_direct(
            B, old_flat, new_flat, nm_arr,
            self._pos_np, gs_arr, ge_arr,
            cl._nc, cl._inv_cs, cl.half_box,
            cl._neighbor_offsets, cl._sorted_order,
            cl._cell_starts, cl._cell_counts,
            self.box, self.inv_box, self.r_rep_sq, self.rep_e,
            max_moved)

        return results.tolist()

    def compute_segment_pair_energy(self, pos_a_np, pos_b_np):
        """Compute pairwise energy between two groups of beads (for rank-1)."""
        return _compute_segment_pair_energy_fast(
            pos_a_np, pos_b_np, self.box, self.inv_box,
            self.r_rep_sq, self.rep_e)
