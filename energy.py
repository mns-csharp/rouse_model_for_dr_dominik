"""
Energy computation: 3-zone excluded-volume kernel with hybrid cell-list.

Optimized per Migacz et al. "Parallel Implementation of a Sequential Markov
Chain in Monte Carlo Simulations":
  - Batched delta-E: FP32 4D broadcast [B, max_moved, n_total, 3] on GPU
  - Batched E_mm: FP32 5D broadcast [B, B, mm, mm, 3] for rank-1 matrices
  - torch.compile kernel fusion eliminates per-op GPU launch overhead
  - Vectorized masking (no Python loops on GPU tensors)
  - Cell-list path retained for sequential CPU mode

All coordinate operations delegate to NumberSpace.
"""

import torch
import math
from .config import SimulationConfig
from .number_space import NumberSpace


# ---------------------------------------------------------------------------
# Compiled GPU kernels for batched energy computation (FP32)
# ---------------------------------------------------------------------------

# 'reduce-overhead' enables CUDA graphs, which require static tensor shapes.
# MC simulations have dynamic shapes (varying segment sizes per sweep),
# causing CUDA graph capture failures (cudaErrorUnknown) after many distinct
# shapes are encountered. Use 'default' for kernel fusion without CUDA graphs.
_COMPILE_MODE = 'default'

def _batched_delta_e_kernel_impl(pos_f32, old_batch, new_batch,
                                  gs_tensor, nm_tensor,
                                  box, inv_box, r_rep_sq, rep_e,
                                  V, max_moved, n_total):
    """
    Compute batched delta-E using 4D broadcast.
    FP32 for throughput (63x faster than FP64 on consumer GPUs).
    Vectorized masking replaces Python loop over proposals.
    """
    # 4D broadcast: [V, max_moved, n_total, 3]
    d_old = pos_f32[None, None, :, :] - old_batch[:, :, None, :]
    d_old = d_old - box * torch.round(d_old * inv_box)
    r2_old = (d_old * d_old).sum(dim=3)  # [V, max_moved, n_total]

    d_new = pos_f32[None, None, :, :] - new_batch[:, :, None, :]
    d_new = d_new - box * torch.round(d_new * inv_box)
    r2_new = (d_new * d_new).sum(dim=3)

    # Vectorized self-interaction mask
    # Build [V, n_total] mask: True for beads that are part of the moved segment
    bead_idx = torch.arange(n_total, device=pos_f32.device).unsqueeze(0)  # [1, n_total]
    self_mask = (bead_idx >= gs_tensor.unsqueeze(1)) & \
                (bead_idx < (gs_tensor + nm_tensor).unsqueeze(1))  # [V, n_total]
    self_mask_3d = self_mask.unsqueeze(1)  # [V, 1, n_total] → broadcast to [V, max_moved, n_total]
    r2_old = r2_old.masked_fill(self_mask_3d, float('inf'))
    r2_new = r2_new.masked_fill(self_mask_3d, float('inf'))

    # Padding mask for proposals with fewer moved beads
    move_idx = torch.arange(max_moved, device=pos_f32.device).unsqueeze(0)  # [1, max_moved]
    pad_mask = move_idx >= nm_tensor.unsqueeze(1)  # [V, max_moved]
    pad_mask_3d = pad_mask.unsqueeze(2)  # [V, max_moved, 1]
    r2_old = r2_old.masked_fill(pad_mask_3d, float('inf'))
    r2_new = r2_new.masked_fill(pad_mask_3d, float('inf'))

    # Count overlaps per proposal: [V]
    e_old_v = (r2_old < r_rep_sq).reshape(V, -1).sum(dim=1)
    e_new_v = (r2_new < r_rep_sq).reshape(V, -1).sum(dim=1)
    delta_v = ((e_new_v.float() - e_old_v.float()) * rep_e).tolist()

    return delta_v


def _batched_emm_correction_kernel_impl(old_batch, new_batch,
                                         pair_i, pair_j,
                                         box, inv_box, r_rep_sq, rep_e, B):
    """
    Fused correction kernel: compute (E11-E01)-(E10-E00) in one pass.

    Instead of 4 separate 5D broadcasts, loads all 4 position combinations
    once and computes the correction directly. Returns a single [B, B]
    matrix instead of four, reducing GPU memory and D2H transfer by 4×.

    Only computes for sparse (pair_i, pair_j) pairs that pass the
    bounding-sphere proximity filter — skips distant pairs entirely.
    """
    n_pairs = pair_i.shape[0]
    if n_pairs == 0:
        return torch.zeros(B, B, dtype=torch.float32, device=old_batch.device)

    # Gather positions for valid pairs: [n_pairs, mm, 3]
    oi = old_batch[pair_i]   # [P, mm, 3]
    ni = new_batch[pair_i]
    oj = old_batch[pair_j]
    nj = new_batch[pair_j]

    # 4D broadcast: [P, mm_i, mm_j, 3]
    oi_e = oi[:, :, None, :]   # [P, mm, 1, 3]
    ni_e = ni[:, :, None, :]
    oj_e = oj[:, None, :, :]   # [P, 1, mm, 3]
    nj_e = nj[:, None, :, :]

    # E00: old_i vs old_j
    d = oi_e - oj_e
    d = d - box * torch.round(d * inv_box)
    c00 = ((d * d).sum(-1) < r_rep_sq).sum(dim=(1, 2))

    # E01: old_i vs new_j
    d = oi_e - nj_e
    d = d - box * torch.round(d * inv_box)
    c01 = ((d * d).sum(-1) < r_rep_sq).sum(dim=(1, 2))

    # E10: new_i vs old_j
    d = ni_e - oj_e
    d = d - box * torch.round(d * inv_box)
    c10 = ((d * d).sum(-1) < r_rep_sq).sum(dim=(1, 2))

    # E11: new_i vs new_j
    d = ni_e - nj_e
    d = d - box * torch.round(d * inv_box)
    c11 = ((d * d).sum(-1) < r_rep_sq).sum(dim=(1, 2))

    # Correction = (E11 - E01) - (E10 - E00)
    corr_vals = ((c11.float() - c01.float()) - (c10.float() - c00.float())) * rep_e

    # Scatter into [B, B] matrix
    corr = torch.zeros(B, B, dtype=torch.float32, device=old_batch.device)
    corr[pair_i, pair_j] = corr_vals
    return corr


# Try to compile; fall back gracefully if torch.compile is unavailable
try:
    _batched_delta_e_kernel = torch.compile(
        _batched_delta_e_kernel_impl, mode=_COMPILE_MODE, dynamic=True)
    _batched_emm_correction_kernel = torch.compile(
        _batched_emm_correction_kernel_impl, mode=_COMPILE_MODE, dynamic=True)
except Exception:
    _batched_delta_e_kernel = _batched_delta_e_kernel_impl
    _batched_emm_correction_kernel = _batched_emm_correction_kernel_impl


def energy_kernel(r2: torch.Tensor, r_rep_sq: float, r_max_sq: float,
                  repulsive_energy: float, contact_energy: float) -> torch.Tensor:
    """Three-zone contact kernel evaluated on squared distances."""
    e = torch.zeros_like(r2)
    repulsive_mask = r2 < r_rep_sq
    e[repulsive_mask] = repulsive_energy
    if contact_energy != 0.0:
        contact_mask = (r2 >= r_rep_sq) & (r2 < r_max_sq)
        e[contact_mask] = contact_energy
    return e


def compute_segment_pair_energy(pos_a: torch.Tensor, pos_b: torch.Tensor,
                                 ns: 'NumberSpace', r_rep_sq: float,
                                 repulsive_energy: float) -> float:
    """
    Compute total excluded-volume energy between two groups of beads.
    Fully vectorized with early exit.
    """
    delta = pos_b.unsqueeze(0) - pos_a.unsqueeze(1)
    delta = delta - ns.box_size * torch.round(delta * ns._inv_box)
    r2 = (delta * delta).sum(dim=2)
    overlap = r2 < r_rep_sq
    if not overlap.any():
        return 0.0
    return overlap.sum().item() * repulsive_energy


class CellList:
    """
    Hybrid cell list: numpy-vectorized build + numpy-accelerated gather.

    Build: numpy argsort for cell assignment → flat sorted arrays with
    offset table. No Python per-bead loop.
    Gather: numpy concatenation of cell slices + boolean mask for exclusion.
    """

    def __init__(self, cfg: SimulationConfig, ns: NumberSpace):
        import numpy as np
        self.ns = ns
        self.r_max = cfg.r_max
        nc = max(3, int(math.floor(cfg.box_size / cfg.r_max)))
        self.n_cells = nc
        self.cell_size = cfg.box_size / nc
        self._half_box = ns.half_box
        self._inv_cs = 1.0 / self.cell_size
        self._nc = nc

        # Pre-compute neighbor cell list as Python list-of-lists (fast lookup)
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

        # Populated by build(): numpy flat arrays for vectorized gather
        self._sorted_order = np.empty(0, dtype=np.int64)
        self._cell_starts = np.zeros(nc3, dtype=np.int64)
        self._cell_counts = np.zeros(nc3, dtype=np.int64)

    def build(self, positions_cpu: torch.Tensor):
        """
        Build cell list from flat positions [total_beads, 3] (CPU tensor).

        Fully vectorized: numpy cell assignment + bincount + argsort.
        Zero Python per-bead iteration.
        """
        import numpy as np
        nc = self._nc
        inv_cs = self._inv_cs
        half_box = self._half_box
        nc_m1 = nc - 1
        nc3 = self._nc3

        # Vectorized cell assignment (zero-copy numpy view)
        pos_np = positions_cpu.numpy()
        cx = np.clip(((pos_np[:, 0] + half_box) * inv_cs).astype(np.int64), 0, nc_m1)
        cy = np.clip(((pos_np[:, 1] + half_box) * inv_cs).astype(np.int64), 0, nc_m1)
        cz = np.clip(((pos_np[:, 2] + half_box) * inv_cs).astype(np.int64), 0, nc_m1)
        cell_idx = (cx * nc + cy) * nc + cz

        # Sort atom indices by cell (numpy — no Python loop)
        self._sorted_order = np.argsort(cell_idx, kind='mergesort')

        # Cell offsets via bincount + cumsum
        counts = np.bincount(cell_idx, minlength=nc3).astype(np.int64)
        starts = np.empty(nc3, dtype=np.int64)
        starts[0] = 0
        np.cumsum(counts[:-1], out=starts[1:])

        self._cell_counts = counts
        self._cell_starts = starts

    def gather_neighbors(self, old_positions, new_positions,
                          global_exclude_start: int,
                          global_exclude_end: int) -> list:
        """
        Gather neighbor atom indices for moved beads (old + new positions).

        Uses numpy arrays for cell lookups and bulk concatenation,
        then Python filtering for exclusion (small result set).

        Returns: Python list of atom indices (to be used with torch indexing).
        """
        import numpy as np
        nc = self._nc
        inv_cs = self._inv_cs
        half_box = self._half_box
        neighbor_cells = self._neighbor_cells
        nc_m1 = nc - 1
        sorted_order = self._sorted_order
        cell_starts = self._cell_starts
        cell_counts = self._cell_counts

        # Convert to Python lists once (avoids per-bead .item() overhead)
        old_list = old_positions.tolist()
        new_list = new_positions.tolist()

        # Find cells occupied by moved beads (pure Python scalar math)
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

        # Expand to neighbor cells
        nbr_cells = set()
        for qc in query_cells:
            nbr_cells.update(neighbor_cells[qc])

        # Gather atom indices via numpy slices (bulk concatenation)
        slices = []
        for c in nbr_cells:
            cnt = cell_counts[c]
            if cnt > 0:
                s = cell_starts[c]
                slices.append(sorted_order[s:s + cnt])

        if not slices:
            return []

        all_atoms = np.concatenate(slices)

        # Exclude moved beads (numpy boolean mask — fast for small exclusion range)
        gs = global_exclude_start
        ge = global_exclude_end
        mask = (all_atoms < gs) | (all_atoms >= ge)
        return all_atoms[mask].tolist()


class EnergyComputer:
    """
    Computes excluded-volume energy and delta-E for MC move proposals.

    Uses hybrid cell-list: torch build + Python gather + torch distances.
    """

    def __init__(self, cfg: SimulationConfig, ns: NumberSpace):
        self.cfg = cfg
        self.ns = ns
        self.cell_list = CellList(cfg, ns)
        self.device = cfg.get_torch_device()
        self.dtype = cfg.dtype
        self.r_rep_sq = cfg.r_rep_sq
        self.r_max_sq = cfg.r_max_sq
        self.repulsive_energy = cfg.repulsive_energy
        self.contact_energy = cfg.contact_energy
        self.kBT = cfg.kBT
        # Cached CPU positions for delta-E distance computation
        self._pos_cpu = None

    def rebuild_cell_list(self, positions_flat: torch.Tensor):
        """Rebuild cell list and cache CPU positions."""
        if positions_flat.device.type != 'cpu':
            pos_cpu = positions_flat.detach().cpu()
        else:
            # Use a view — auto-updates when state.apply_move() modifies positions.
            # The stale cell list may miss a few moved atoms but found atoms
            # have correct positions, keeping energy computation accurate.
            pos_cpu = positions_flat.detach()
        self._pos_cpu = pos_cpu
        self.cell_list.build(pos_cpu)

    def compute_delta_energy_move(self, positions_flat: torch.Tensor,
                                   chain_idx: int, bead_start: int,
                                   old_positions: torch.Tensor,
                                   new_positions: torch.Tensor,
                                   N: int) -> float:
        """
        Compute ΔE for a proposed move of a contiguous segment.

        Python cell-list gather (~50μs) + PyTorch distance comp (~250μs).
        """
        n_moved = old_positions.shape[0]
        global_start = chain_idx * N + bead_start
        global_end = global_start + n_moved

        # CPU tensors for cell lookup + distance computation
        old_cpu = (old_positions.detach().cpu()
                   if old_positions.device.type != 'cpu'
                   else old_positions)
        new_cpu = (new_positions.detach().cpu()
                   if new_positions.device.type != 'cpu'
                   else new_positions)

        # Python-fast neighbor gathering
        stationary_list = self.cell_list.gather_neighbors(
            old_cpu, new_cpu, global_start, global_end)

        delta_e = 0.0

        # Part 1: moved vs stationary — small pairwise via PyTorch
        if stationary_list:
            stationary_indices = torch.tensor(stationary_list, dtype=torch.long)
            stationary_pos = self._pos_cpu[stationary_indices]  # [K, 3]

            box = self.ns.box_size
            inv_box = self.ns._inv_box

            # Old: [n_moved, K, 3]
            d_old = stationary_pos.unsqueeze(0) - old_cpu.unsqueeze(1)
            d_old = d_old - box * torch.round(d_old * inv_box)
            r2_old = (d_old * d_old).sum(dim=2)

            # New: [n_moved, K, 3]
            d_new = stationary_pos.unsqueeze(0) - new_cpu.unsqueeze(1)
            d_new = d_new - box * torch.round(d_new * inv_box)
            r2_new = (d_new * d_new).sum(dim=2)

            e_old = (r2_old < self.r_rep_sq).sum().item()
            e_new = (r2_new < self.r_rep_sq).sum().item()
            delta_e += (e_new - e_old) * self.repulsive_energy

        # Part 2: intra-segment (moved-moved, upper triangle only)
        if n_moved > 1:
            box = self.ns.box_size
            inv_box = self.ns._inv_box

            d_old_mm = old_cpu.unsqueeze(1) - old_cpu.unsqueeze(0)
            d_old_mm = d_old_mm - box * torch.round(d_old_mm * inv_box)
            r2_old_mm = (d_old_mm * d_old_mm).sum(dim=2)

            d_new_mm = new_cpu.unsqueeze(1) - new_cpu.unsqueeze(0)
            d_new_mm = d_new_mm - box * torch.round(d_new_mm * inv_box)
            r2_new_mm = (d_new_mm * d_new_mm).sum(dim=2)

            triu_mask = torch.triu(
                torch.ones(n_moved, n_moved, dtype=torch.bool), diagonal=1)
            e_old_mm = (r2_old_mm[triu_mask] < self.r_rep_sq).sum().item()
            e_new_mm = (r2_new_mm[triu_mask] < self.r_rep_sq).sum().item()
            delta_e += (e_new_mm - e_old_mm) * self.repulsive_energy

        return delta_e

    def compute_batch_delta_energy(self, positions_flat: torch.Tensor,
                                    proposals, N: int) -> list:
        """
        Compute delta-E for all B proposals in a batch using GPU-parallel
        direct pairwise. One [B, max_moved, n_total, 3] kernel instead of
        B sequential calls.

        Accepts either list[MoveProposal] or BatchProposal.
        FP32 computation + vectorized masking (no Python loops on GPU).
        """
        from .batch_proposal import BatchProposal

        if isinstance(proposals, BatchProposal):
            return self._batch_delta_e_fused(positions_flat, proposals, N)

        B = len(proposals)
        results = [0.0] * B
        valid_indices = [i for i in range(B) if proposals[i].n_moved > 0]
        if not valid_indices:
            return results

        V = len(valid_indices)
        device = positions_flat.device
        n_total = positions_flat.shape[0]

        pos_f32 = positions_flat.float() if positions_flat.dtype != torch.float32 else positions_flat
        old_batch = torch.zeros(V, max(proposals[i].n_moved for i in valid_indices), 3,
                                dtype=torch.float32, device=device)
        new_batch = torch.zeros_like(old_batch)
        gs_tensor = torch.empty(V, dtype=torch.long, device=device)
        nm_tensor = torch.empty(V, dtype=torch.long, device=device)

        for vi, idx in enumerate(valid_indices):
            p = proposals[idx]
            nm = p.n_moved
            old_batch[vi, :nm] = p.old_positions.float() if p.old_positions.dtype != torch.float32 else p.old_positions
            new_batch[vi, :nm] = p.new_positions.float() if p.new_positions.dtype != torch.float32 else p.new_positions
            gs_tensor[vi] = p.chain_idx * N + p.bead_start
            nm_tensor[vi] = nm

        delta_v = _batched_delta_e_kernel(
            pos_f32, old_batch, new_batch, gs_tensor, nm_tensor,
            self.ns.box_size, self.ns._inv_box, self.r_rep_sq,
            self.repulsive_energy, V, old_batch.shape[1], n_total)

        for vi, idx in enumerate(valid_indices):
            results[idx] = delta_v[vi]
        return results

    def _batch_delta_e_fused(self, positions_flat, bp, N):
        """Fast path: BatchProposal already has GPU batch tensors."""
        B = bp.B
        device = positions_flat.device
        n_total = positions_flat.shape[0]
        results = [0.0] * B

        valid_indices = [i for i in range(B) if bp.n_moved[i] > 0]
        if not valid_indices:
            return results

        V = len(valid_indices)
        old_f32, new_f32 = bp.get_f32()

        # If all proposals are valid (common case), skip re-indexing
        if V == B:
            pos_f32 = positions_flat.float() if positions_flat.dtype != torch.float32 else positions_flat
            gs, nm = bp.get_gs_nm_tensors(N, device)
            delta_v = _batched_delta_e_kernel(
                pos_f32, old_f32, new_f32, gs, nm,
                self.ns.box_size, self.ns._inv_box, self.r_rep_sq,
                self.repulsive_energy, V, bp.max_moved, n_total)
            return delta_v

        # Subset valid proposals
        pos_f32 = positions_flat.float() if positions_flat.dtype != torch.float32 else positions_flat
        vi_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
        sub_old = old_f32[vi_tensor]
        sub_new = new_f32[vi_tensor]
        gs = torch.tensor([bp.chain_idx[i] * N + bp.bead_start[i] for i in valid_indices],
                          dtype=torch.long, device=device)
        nm = torch.tensor([bp.n_moved[i] for i in valid_indices],
                          dtype=torch.long, device=device)

        delta_v = _batched_delta_e_kernel(
            pos_f32, sub_old, sub_new, gs, nm,
            self.ns.box_size, self.ns._inv_box, self.r_rep_sq,
            self.repulsive_energy, V, bp.max_moved, n_total)

        for vi, idx in enumerate(valid_indices):
            results[idx] = delta_v[vi]
        return results

    def compute_batch_energy_matrices(self, positions_flat: torch.Tensor,
                                       proposals, N: int):
        """
        Pre-compute ALL energy data for a batch in parallel on GPU:
          1. E_total[i]: delta-E of each proposal vs stationary system
          2. correction[i,j]: rank-1 correction matrix = (E11-E01)-(E10-E00)

        Optimizations over the original 4-matrix approach:
          - Fused kernel: computes correction directly in one pass (4× less output)
          - Bounding-sphere filtering: skips distant pairs that cannot interact
          - Single D2H transfer: 1 matrix instead of 4

        Core parallelization from Migacz et al. "Parallel Implementation of
        a Sequential Markov Chain in Monte Carlo Simulations."
        Accepts either list[MoveProposal] or BatchProposal.

        Returns:
            delta_e: list[float] of length B
            correction: [B, B] tensor on CPU (rank-1 correction matrix)
        """
        from .batch_proposal import BatchProposal

        is_bp = isinstance(proposals, BatchProposal)
        B = proposals.B if is_bp else len(proposals)

        # --- E_total (batched delta-E) ---
        delta_e = self.compute_batch_delta_energy(positions_flat, proposals, N)

        # --- Correction matrix: fused (E11-E01)-(E10-E00) ---
        device = positions_flat.device
        box = self.ns.box_size
        inv_box = self.ns._inv_box
        r_rep_sq = self.r_rep_sq
        rep_e = self.repulsive_energy

        if is_bp:
            old_f32, new_f32 = proposals.get_f32()
            nm_list = proposals.n_moved
        else:
            valid = [i for i in range(B) if proposals[i].n_moved > 0]
            if len(valid) < 2:
                return delta_e, torch.zeros(B, B)

            max_moved = max(proposals[i].n_moved for i in valid)
            old_f32 = torch.zeros(B, max_moved, 3, dtype=torch.float32, device=device)
            new_f32 = torch.zeros(B, max_moved, 3, dtype=torch.float32, device=device)
            nm_list = [0] * B
            for i in valid:
                p = proposals[i]
                nm = p.n_moved
                old_f32[i, :nm] = p.old_positions.float() if p.old_positions.dtype != torch.float32 else p.old_positions
                new_f32[i, :nm] = p.new_positions.float() if p.new_positions.dtype != torch.float32 else p.new_positions
                nm_list[i] = nm

        valid_indices = [i for i, nm in enumerate(nm_list) if nm > 0]
        if len(valid_indices) < 2:
            return delta_e, torch.zeros(B, B)

        # --- Bounding-sphere pair filtering ---
        # Compute centroid and max radius for each proposal's moved beads
        # (old + new combined). Skip pairs whose bounding spheres can't overlap.
        r_rep = math.sqrt(r_rep_sq)
        centroids = torch.zeros(B, 3, dtype=torch.float32, device=device)
        radii = torch.zeros(B, dtype=torch.float32, device=device)

        for i in valid_indices:
            nm = nm_list[i]
            # Combine old + new positions for bounding sphere
            all_pos = torch.cat([old_f32[i, :nm], new_f32[i, :nm]], dim=0)  # [2*nm, 3]
            centroid = all_pos.mean(dim=0)
            centroids[i] = centroid
            # Max distance from centroid to any bead
            d = all_pos - centroid.unsqueeze(0)
            d = d - box * torch.round(d * inv_box)
            radii[i] = (d * d).sum(dim=1).max().sqrt()

        # Build sparse pair list: only pairs whose spheres could overlap
        pair_i_list = []
        pair_j_list = []
        for ii in range(len(valid_indices)):
            i = valid_indices[ii]
            for jj in range(ii + 1, len(valid_indices)):
                j = valid_indices[jj]
                # MIC distance between centroids
                dc = centroids[i] - centroids[j]
                dc = dc - box * torch.round(dc * inv_box)
                dist = (dc * dc).sum().sqrt().item()
                # Can interact only if dist < radius_i + radius_j + r_rep
                if dist < radii[i].item() + radii[j].item() + r_rep:
                    pair_i_list.append(i)
                    pair_j_list.append(j)

        if not pair_i_list:
            return delta_e, torch.zeros(B, B)

        pair_i_t = torch.tensor(pair_i_list, dtype=torch.long, device=device)
        pair_j_t = torch.tensor(pair_j_list, dtype=torch.long, device=device)

        # Fused correction kernel: single pass, single output matrix
        correction = _batched_emm_correction_kernel(
            old_f32, new_f32, pair_i_t, pair_j_t,
            box, inv_box, r_rep_sq, rep_e, B)

        # Move to CPU for fast scalar access in acceptance loop
        return delta_e, correction.cpu()


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

def compute_delta_energy_bruteforce(positions_flat: torch.Tensor,
                                     chain_idx: int, bead_start: int,
                                     old_positions: torch.Tensor,
                                     new_positions: torch.Tensor,
                                     N: int, ns: 'NumberSpace',
                                     r_rep_sq: float,
                                     repulsive_energy: float) -> float:
    """
    Brute-force all-pairs delta-E computation (no cell list).

    Computes energy change by scanning ALL beads in the system against the
    moved segment. Used as a reference for validating the cell-list path.
    O(n_moved * n_total) — too slow for production, correct by construction.
    """
    n_moved = old_positions.shape[0]
    n_total = positions_flat.shape[0]
    global_start = chain_idx * N + bead_start
    global_end = global_start + n_moved
    box = ns.box_size
    inv_box = ns._inv_box

    # Stationary beads = all beads NOT in the moved segment
    mask = torch.ones(n_total, dtype=torch.bool, device=positions_flat.device)
    mask[global_start:global_end] = False
    stationary = positions_flat[mask]  # [K, 3]

    delta_e = 0.0

    if stationary.shape[0] > 0:
        # Old interactions
        d_old = stationary.unsqueeze(0) - old_positions.unsqueeze(1)
        d_old = d_old - box * torch.round(d_old * inv_box)
        r2_old = (d_old * d_old).sum(dim=2)
        e_old = (r2_old < r_rep_sq).sum().item()

        # New interactions
        d_new = stationary.unsqueeze(0) - new_positions.unsqueeze(1)
        d_new = d_new - box * torch.round(d_new * inv_box)
        r2_new = (d_new * d_new).sum(dim=2)
        e_new = (r2_new < r_rep_sq).sum().item()

        delta_e += (e_new - e_old) * repulsive_energy

    # Intra-segment (moved-moved) upper triangle
    if n_moved > 1:
        d_old_mm = old_positions.unsqueeze(1) - old_positions.unsqueeze(0)
        d_old_mm = d_old_mm - box * torch.round(d_old_mm * inv_box)
        r2_old_mm = (d_old_mm * d_old_mm).sum(dim=2)

        d_new_mm = new_positions.unsqueeze(1) - new_positions.unsqueeze(0)
        d_new_mm = d_new_mm - box * torch.round(d_new_mm * inv_box)
        r2_new_mm = (d_new_mm * d_new_mm).sum(dim=2)

        triu = torch.triu(torch.ones(n_moved, n_moved, dtype=torch.bool,
                                      device=positions_flat.device), diagonal=1)
        e_old_mm = (r2_old_mm[triu] < r_rep_sq).sum().item()
        e_new_mm = (r2_new_mm[triu] < r_rep_sq).sum().item()
        delta_e += (e_new_mm - e_old_mm) * repulsive_energy

    return delta_e


def validate_cell_list_vs_bruteforce(energy_comp: 'EnergyComputer',
                                      state: 'ChainState',
                                      cfg: 'SimulationConfig',
                                      n_samples: int = 10) -> dict:
    """
    Validate that cell-list delta-E matches brute-force all-pairs.

    Proposes n_samples random hinge moves and compares the two energy paths.
    Returns dict with 'max_abs_error', 'all_match' (bool), and 'details'.
    """
    from .mc_moves import propose_segment_move
    ns = state.ns
    N = cfg.N
    gen = cfg.get_torch_gen()
    positions_flat = state.get_all_flat()
    energy_comp.rebuild_cell_list(positions_flat)

    max_err = 0.0
    details = []
    seg_info = state.segments

    for i in range(n_samples):
        chain_idx = i % cfg.n_chains
        local_seg = i % seg_info.segs_per_chain
        proposal = propose_segment_move(state, chain_idx, local_seg, gen, cfg)
        if proposal.n_moved == 0:
            continue

        de_cell = energy_comp.compute_delta_energy_move(
            positions_flat, proposal.chain_idx, proposal.bead_start,
            proposal.old_positions, proposal.new_positions, N)

        de_brute = compute_delta_energy_bruteforce(
            positions_flat, proposal.chain_idx, proposal.bead_start,
            proposal.old_positions, proposal.new_positions,
            N, ns, cfg.r_rep_sq, cfg.repulsive_energy)

        err = abs(de_cell - de_brute)
        max_err = max(max_err, err)
        details.append({'cell_list': de_cell, 'bruteforce': de_brute, 'error': err})

    return {
        'max_abs_error': max_err,
        'all_match': max_err == 0.0,
        'details': details,
    }


def validate_fp32_vs_fp64(energy_comp: 'EnergyComputer',
                           state: 'ChainState',
                           cfg: 'SimulationConfig',
                           n_samples: int = 10,
                           atol: float = 1.0) -> dict:
    """
    Validate that FP32 GPU batched delta-E agrees with FP64 CPU delta-E.

    Proposes n_samples moves and compares:
      - FP64 CPU: cell-list based compute_delta_energy_move (float64)
      - FP32 GPU: batched compute_batch_delta_energy (float32 kernel)

    For the athermal overlap-counting kernel, differences should be exactly
    zero (boolean comparisons are precision-independent). The atol parameter
    guards against edge cases near the cutoff boundary.

    Returns dict with 'max_abs_error', 'all_within_tol', and 'details'.
    """
    from .mc_moves import propose_segment_move
    N = cfg.N
    gen = cfg.get_torch_gen()
    positions_flat = state.get_all_flat()
    energy_comp.rebuild_cell_list(positions_flat)

    max_err = 0.0
    details = []
    seg_info = state.segments

    for i in range(n_samples):
        chain_idx = i % cfg.n_chains
        local_seg = i % seg_info.segs_per_chain
        proposal = propose_segment_move(state, chain_idx, local_seg, gen, cfg)
        if proposal.n_moved == 0:
            continue

        # FP64 CPU path (cell-list)
        de_fp64 = energy_comp.compute_delta_energy_move(
            positions_flat, proposal.chain_idx, proposal.bead_start,
            proposal.old_positions, proposal.new_positions, N)

        # FP32 batched path (uses float32 kernel internally)
        de_fp32_list = energy_comp.compute_batch_delta_energy(
            positions_flat, [proposal], N)
        de_fp32 = de_fp32_list[0]

        err = abs(de_fp64 - de_fp32)
        max_err = max(max_err, err)
        details.append({'fp64': de_fp64, 'fp32': de_fp32, 'error': err})

    return {
        'max_abs_error': max_err,
        'all_within_tol': max_err <= atol,
        'details': details,
    }
