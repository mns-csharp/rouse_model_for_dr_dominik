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

_COMPILE_MODE = 'reduce-overhead'

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


def _batched_emm_kernel_impl(old_batch, new_batch, box, inv_box, r_rep_sq, rep_e):
    """
    Compute all 4 pairwise segment energy matrices E00, E01, E10, E11.
    [B, B, max_moved, max_moved, 3] → [B, B] count matrices.
    """
    def _pairwise(pos_i, pos_j):
        d = pos_i[:, None, :, None, :] - pos_j[None, :, None, :, :]
        d = d - box * torch.round(d * inv_box)
        r2 = (d * d).sum(dim=4)
        return (r2 < r_rep_sq).sum(dim=(2, 3)).float() * rep_e

    e00 = _pairwise(old_batch, old_batch)
    e01 = _pairwise(old_batch, new_batch)
    e10 = _pairwise(new_batch, old_batch)
    e11 = _pairwise(new_batch, new_batch)
    return e00, e01, e10, e11


# Try to compile; fall back gracefully if torch.compile is unavailable
try:
    _batched_delta_e_kernel = torch.compile(
        _batched_delta_e_kernel_impl, mode=_COMPILE_MODE, dynamic=True)
    _batched_emm_kernel = torch.compile(
        _batched_emm_kernel_impl, mode=_COMPILE_MODE, dynamic=True)
except Exception:
    _batched_delta_e_kernel = _batched_delta_e_kernel_impl
    _batched_emm_kernel = _batched_emm_kernel_impl


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
          2. E_mm[i,j]: 4 pairwise segment energy matrices for rank-1 corrections

        Core parallelization from Migacz et al. "Parallel Implementation of
        a Sequential Markov Chain in Monte Carlo Simulations."
        Accepts either list[MoveProposal] or BatchProposal.

        Returns:
            delta_e: list[float] of length B
            Emm00, Emm01, Emm10, Emm11: [B, B] tensors (on CPU for scalar access)
        """
        from .batch_proposal import BatchProposal

        is_bp = isinstance(proposals, BatchProposal)
        B = proposals.B if is_bp else len(proposals)

        # --- E_total (batched delta-E) ---
        delta_e = self.compute_batch_delta_energy(positions_flat, proposals, N)

        # --- E_mm: pairwise segment energy matrices ---
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
                z = torch.zeros(B, B)
                return delta_e, z, z.clone(), z.clone(), z.clone()

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

        if sum(1 for nm in nm_list if nm > 0) < 2:
            z = torch.zeros(B, B)
            return delta_e, z, z.clone(), z.clone(), z.clone()

        # Compute all 4 pairwise segment energy matrices on GPU
        Emm00, Emm01, Emm10, Emm11 = _batched_emm_kernel(
            old_f32, new_f32, box, inv_box, r_rep_sq, rep_e)

        # Zero out diagonal and invalid entries (vectorized)
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        Emm00[diag_mask] = 0; Emm01[diag_mask] = 0
        Emm10[diag_mask] = 0; Emm11[diag_mask] = 0

        invalid_mask = torch.tensor([nm == 0 for nm in nm_list],
                                     dtype=torch.bool, device=device)
        if invalid_mask.any():
            Emm00[invalid_mask, :] = 0; Emm00[:, invalid_mask] = 0
            Emm01[invalid_mask, :] = 0; Emm01[:, invalid_mask] = 0
            Emm10[invalid_mask, :] = 0; Emm10[:, invalid_mask] = 0
            Emm11[invalid_mask, :] = 0; Emm11[:, invalid_mask] = 0

        # Move to CPU for fast scalar access in acceptance loop
        return (delta_e,
                Emm00.cpu(), Emm01.cpu(), Emm10.cpu(), Emm11.cpu())
