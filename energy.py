"""
Energy computation: 3-zone excluded-volume kernel with hybrid cell-list.

Optimized for minimal per-call overhead:
  - Cell-list build: torch.argsort (vectorized) → Python dict (O(1) lookup)
  - Neighbor gathering: Python set operations (~50μs vs ~600μs with torch.unique)
  - Distance computation: PyTorch tensors on [n_moved, ~K_neighbors] (small)
  - Cell-list rebuild uses cached CPU positions for instant distance lookups

All coordinate operations delegate to NumberSpace.
"""

import torch
import math
from .config import SimulationConfig
from .number_space import NumberSpace


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
    Hybrid cell list: torch-vectorized build + Python-dict O(1) gather.

    Build: torch.argsort for cell assignment, then .tolist() → Python dict.
    Gather: Python set operations for cell lookup (~50μs per call vs ~600μs).
    """

    def __init__(self, cfg: SimulationConfig, ns: NumberSpace):
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

        # Populated by build()
        self._cells = {}  # {cell_linear_idx: [atom_idx, ...]}

    def build(self, positions_cpu: torch.Tensor):
        """
        Build cell list from flat positions [total_beads, 3] (CPU tensor).

        Uses numpy (zero-copy view) for vectorized cell assignment,
        then builds Python dict from 1D int list.
        """
        import numpy as np
        nc = self._nc
        inv_cs = self._inv_cs
        half_box = self._half_box
        nc_m1 = nc - 1

        # Zero-copy numpy view for vectorized cell assignment
        pos_np = positions_cpu.numpy()
        cx = np.clip(((pos_np[:, 0] + half_box) * inv_cs).astype(np.int64), 0, nc_m1)
        cy = np.clip(((pos_np[:, 1] + half_box) * inv_cs).astype(np.int64), 0, nc_m1)
        cz = np.clip(((pos_np[:, 2] + half_box) * inv_cs).astype(np.int64), 0, nc_m1)
        cell_idx_list = ((cx * nc + cy) * nc + cz).tolist()  # 1D int list

        # Build Python dict (O(n) single pass)
        cells = {}
        for i, c in enumerate(cell_idx_list):
            if c in cells:
                cells[c].append(i)
            else:
                cells[c] = [i]
        self._cells = cells

    def gather_neighbors(self, old_positions, new_positions,
                          global_exclude_start: int,
                          global_exclude_end: int) -> list:
        """
        Gather neighbor atom indices for moved beads (old + new positions).
        Pure Python set operations for minimal overhead.

        Returns: Python list of atom indices (to be used with torch indexing).
        """
        nc = self._nc
        inv_cs = self._inv_cs
        half_box = self._half_box
        cells = self._cells
        neighbor_cells = self._neighbor_cells
        nc_m1 = nc - 1

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

        # Collect neighbor cells (Python set for dedup)
        nbr_cells = set()
        for qc in query_cells:
            nbr_cells.update(neighbor_cells[qc])

        # Gather atom indices, excluding moved beads
        gs = global_exclude_start
        ge = global_exclude_end
        result = []
        for c in nbr_cells:
            atom_list = cells.get(c)
            if atom_list is not None:
                for a in atom_list:
                    if a < gs or a >= ge:
                        result.append(a)

        return result


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

        Intra-segment energy is skipped: all segment/pivot moves are
        rigid-body rotations that preserve intra-segment distances.
        """
        B = len(proposals)
        results = [0.0] * B

        valid_indices = [i for i in range(B) if proposals[i].n_moved > 0]
        if not valid_indices:
            return results

        V = len(valid_indices)
        device = positions_flat.device
        dtype = positions_flat.dtype
        n_total = positions_flat.shape[0]
        box = self.ns.box_size
        inv_box = self.ns._inv_box
        r_rep_sq = self.r_rep_sq

        max_moved = max(proposals[i].n_moved for i in valid_indices)
        old_batch = torch.zeros(V, max_moved, 3, dtype=dtype, device=device)
        new_batch = torch.zeros(V, max_moved, 3, dtype=dtype, device=device)

        # Collect moved-bead global ranges for masking
        gs_list = []
        nm_list = []
        for vi, idx in enumerate(valid_indices):
            p = proposals[idx]
            nm = p.n_moved
            old_batch[vi, :nm] = p.old_positions
            new_batch[vi, :nm] = p.new_positions
            gs_list.append(p.chain_idx * N + p.bead_start)
            nm_list.append(nm)

        # Full 4D broadcast: [V, max_moved, n_total, 3]
        # Fewer kernel launches than dimension-wise (2 kernels vs 12)
        d_old = positions_flat[None, None, :, :] - old_batch[:, :, None, :]
        d_old = d_old - box * torch.round(d_old * inv_box)
        r2_old = (d_old * d_old).sum(dim=3)  # [V, max_moved, n_total]
        del d_old

        d_new = positions_flat[None, None, :, :] - new_batch[:, :, None, :]
        d_new = d_new - box * torch.round(d_new * inv_box)
        r2_new = (d_new * d_new).sum(dim=3)
        del d_new

        # Mask out self-interactions and padding (set to inf)
        for vi in range(V):
            gs = gs_list[vi]
            nm = nm_list[vi]
            r2_old[vi, :, gs:gs + nm] = float('inf')
            r2_new[vi, :, gs:gs + nm] = float('inf')
            if nm < max_moved:
                r2_old[vi, nm:, :] = float('inf')
                r2_new[vi, nm:, :] = float('inf')

        # Count overlaps per proposal: [V]
        e_old_v = (r2_old < r_rep_sq).reshape(V, -1).sum(dim=1)
        e_new_v = (r2_new < r_rep_sq).reshape(V, -1).sum(dim=1)
        delta_v = ((e_new_v.float() - e_old_v.float()) * self.repulsive_energy).tolist()

        for vi, idx in enumerate(valid_indices):
            results[idx] = delta_v[vi]

        return results

    def compute_batch_energy_matrices(self, positions_flat: torch.Tensor,
                                       proposals, N: int):
        """
        Pre-compute ALL energy data for a batch in parallel on GPU:
          1. E_total[i]: delta-E of each proposal vs stationary system
          2. E_mm[i,j]: 4 pairwise segment energy matrices for rank-1 corrections

        This is the core parallelization from the segmented multistep MC paper.
        The sequential acceptance loop then reads pre-computed values —
        no per-pair energy calls needed.

        Returns:
            delta_e: list[float] of length B
            Emm00, Emm01, Emm10, Emm11: [B, B] tensors (on CPU for scalar access)
        """
        B = len(proposals)

        # --- E_total (batched delta-E) ---
        delta_e = self.compute_batch_delta_energy(positions_flat, proposals, N)

        # --- E_mm: pairwise segment energy matrices ---
        device = positions_flat.device
        dtype = positions_flat.dtype
        box = self.ns.box_size
        inv_box = self.ns._inv_box
        r_rep_sq = self.r_rep_sq
        rep_e = self.repulsive_energy

        # Stack proposal positions [B, max_moved, 3]
        valid = [i for i in range(B) if proposals[i].n_moved > 0]
        if len(valid) < 2:
            z = torch.zeros(B, B)
            return delta_e, z, z.clone(), z.clone(), z.clone()

        max_moved = max(proposals[i].n_moved for i in valid)
        old_batch = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
        new_batch = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
        nm_list = [0] * B

        for i in valid:
            p = proposals[i]
            nm = p.n_moved
            old_batch[i, :nm] = p.old_positions
            new_batch[i, :nm] = p.new_positions
            nm_list[i] = nm

        # Compute [B, B, max_moved, max_moved] pairwise distances for all 4 combos.
        # For B=10, max_moved=20: [10,10,20,20,3] = 120K entries = 960KB. Trivial.
        # old_batch[:, None, :, None, :] shape: [B, 1, mm, 1, 3]
        # old_batch[None, :, None, :, :] shape: [1, B, 1, mm, 3]
        # broadcast result: [B, B, mm, mm, 3]

        def _pairwise_energy(pos_i, pos_j):
            d = pos_i[:, None, :, None, :] - pos_j[None, :, None, :, :]
            d = d - box * torch.round(d * inv_box)
            r2 = (d * d).sum(dim=4)  # [B, B, mm, mm]
            return (r2 < r_rep_sq).sum(dim=(2, 3)).float() * rep_e  # [B, B]

        Emm00 = _pairwise_energy(old_batch, old_batch)
        Emm01 = _pairwise_energy(old_batch, new_batch)
        Emm10 = _pairwise_energy(new_batch, old_batch)
        Emm11 = _pairwise_energy(new_batch, new_batch)

        # Zero out diagonal (self-interaction) and invalid entries
        for i in range(B):
            Emm00[i, i] = 0; Emm01[i, i] = 0
            Emm10[i, i] = 0; Emm11[i, i] = 0
        for i in range(B):
            if nm_list[i] == 0:
                Emm00[i, :] = 0; Emm00[:, i] = 0
                Emm01[i, :] = 0; Emm01[:, i] = 0
                Emm10[i, :] = 0; Emm10[:, i] = 0
                Emm11[i, :] = 0; Emm11[:, i] = 0

        # Move to CPU for fast scalar access in acceptance loop
        return (delta_e,
                Emm00.cpu(), Emm01.cpu(), Emm10.cpu(), Emm11.cpu())
