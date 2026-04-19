"""EnergyComputer — cell-list + batched delta-E + batched EMM correction."""

import math
import torch

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.number_space.number_space import NumberSpace
from rouse_model_python.src.libs.energy.cell_list import CellList
from rouse_model_python.src.libs.energy.energy_kernels import EnergyKernels
from rouse_model_python.src.libs.mc_moves.batch_proposal import BatchProposal


class EnergyComputer:
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
        self._pos_cpu = None
        self._triu_cache: dict = {}

    def _get_triu_mask(self, n: int) -> torch.Tensor:
        mask = self._triu_cache.get(n)
        if mask is None:
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            self._triu_cache[n] = mask
        return mask

    def rebuild_cell_list(self, positions_flat: torch.Tensor):
        if positions_flat.device.type != 'cpu':
            pos_cpu = positions_flat.detach().cpu()
        else:
            pos_cpu = positions_flat.detach()
        self._pos_cpu = pos_cpu
        self.cell_list.build(pos_cpu)

    def compute_delta_energy_move(self, positions_flat, chain_idx, bead_start,
                                  old_positions, new_positions, N) -> float:
        n_moved = old_positions.shape[0]
        global_start = chain_idx * N + bead_start
        global_end = global_start + n_moved
        if old_positions.device.type != 'cpu':
            stacked = torch.stack([old_positions, new_positions]).detach().cpu()
            old_cpu = stacked[0]
            new_cpu = stacked[1]
        else:
            old_cpu = old_positions
            new_cpu = new_positions
        stationary_list = self.cell_list.gather_neighbors(
            old_cpu, new_cpu, global_start, global_end)
        delta_overlaps = 0
        if stationary_list:
            stationary_indices = torch.tensor(stationary_list, dtype=torch.long)
            stationary_pos = self._pos_cpu[stationary_indices]
            box = self.ns.box_size
            inv_box = self.ns._inv_box
            d_old = stationary_pos.unsqueeze(0) - old_cpu.unsqueeze(1)
            d_old = d_old - box * torch.round(d_old * inv_box)
            r2_old = (d_old * d_old).sum(dim=2)
            d_new = stationary_pos.unsqueeze(0) - new_cpu.unsqueeze(1)
            d_new = d_new - box * torch.round(d_new * inv_box)
            r2_new = (d_new * d_new).sum(dim=2)
            delta_overlaps = ((r2_new < self.r_rep_sq).sum()
                              - (r2_old < self.r_rep_sq).sum())
        if n_moved > 1:
            box = self.ns.box_size
            inv_box = self.ns._inv_box
            d_old_mm = old_cpu.unsqueeze(1) - old_cpu.unsqueeze(0)
            d_old_mm = d_old_mm - box * torch.round(d_old_mm * inv_box)
            r2_old_mm = (d_old_mm * d_old_mm).sum(dim=2)
            d_new_mm = new_cpu.unsqueeze(1) - new_cpu.unsqueeze(0)
            d_new_mm = d_new_mm - box * torch.round(d_new_mm * inv_box)
            r2_new_mm = (d_new_mm * d_new_mm).sum(dim=2)
            triu_mask = self._get_triu_mask(n_moved)
            delta_overlaps = delta_overlaps + (
                (r2_new_mm[triu_mask] < self.r_rep_sq).sum()
                - (r2_old_mm[triu_mask] < self.r_rep_sq).sum())
        if isinstance(delta_overlaps, int):
            return 0.0
        return float(delta_overlaps.item()) * self.repulsive_energy

    def compute_batch_delta_energy(self, positions_flat, proposals, N):
        if isinstance(proposals, BatchProposal):
            return self._batch_delta_e_fused(positions_flat, proposals, N)
        B = len(proposals)
        device = positions_flat.device
        valid_indices = [i for i in range(B) if proposals[i].n_moved > 0]
        if not valid_indices:
            return torch.zeros(B, dtype=torch.float32, device=device)
        V = len(valid_indices)
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
        kernel = EnergyKernels.delta_e_kernel()
        delta_v = self._delta_e_kernel_tiled(
            kernel, pos_f32, old_batch, new_batch, gs_tensor, nm_tensor,
            V, old_batch.shape[1], n_total)
        results = torch.zeros(B, dtype=torch.float32, device=device)
        vi_t = torch.tensor(valid_indices, dtype=torch.long, device=device)
        results[vi_t] = delta_v
        return results

    def _delta_e_tile_size(self, max_moved: int, n_total: int) -> int:
        """Sub-batch size that keeps the kernel's d_old/d_new intermediates
        within cfg.delta_e_mem_budget_gib. Headroom factor of 4x covers the
        live r2_old/r2_new + boolean masks that torch.compile keeps after
        fusing the diff broadcasts.
        """
        bytes_per_v = max_moved * n_total * 3 * 4  # float32, 3 spatial dims
        budget_bytes = int(self.cfg.delta_e_mem_budget_gib * (1024 ** 3))
        return max(1, budget_bytes // (bytes_per_v * 4))

    def _delta_e_kernel_tiled(self, kernel, pos_f32, sub_old, sub_new,
                              gs, nm, V, max_moved, n_total):
        """Run kernel(...) in tiles of V chosen by _delta_e_tile_size."""
        tile_v = min(V, self._delta_e_tile_size(max_moved, n_total))
        if tile_v >= V:
            return kernel(
                pos_f32, sub_old, sub_new, gs, nm,
                self.ns.box_size, self.ns._inv_box, self.r_rep_sq,
                self.repulsive_energy, V, max_moved, n_total)
        chunks = []
        for s in range(0, V, tile_v):
            e = min(s + tile_v, V)
            chunks.append(kernel(
                pos_f32, sub_old[s:e], sub_new[s:e], gs[s:e], nm[s:e],
                self.ns.box_size, self.ns._inv_box, self.r_rep_sq,
                self.repulsive_energy, e - s, max_moved, n_total))
        return torch.cat(chunks, dim=0)

    def _batch_delta_e_fused(self, positions_flat, bp, N):
        B = bp.B
        device = positions_flat.device
        n_total = positions_flat.shape[0]
        valid_indices = [i for i in range(B) if bp.n_moved[i] > 0]
        if not valid_indices:
            return torch.zeros(B, dtype=torch.float32, device=device)
        V = len(valid_indices)
        old_f32, new_f32 = bp.get_f32()
        kernel = EnergyKernels.delta_e_kernel()
        if V == B:
            pos_f32 = positions_flat.float() if positions_flat.dtype != torch.float32 else positions_flat
            gs, nm = bp.get_gs_nm_tensors(N, device)
            return self._delta_e_kernel_tiled(
                kernel, pos_f32, old_f32, new_f32, gs, nm,
                V, bp.max_moved, n_total)
        pos_f32 = positions_flat.float() if positions_flat.dtype != torch.float32 else positions_flat
        vi_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
        sub_old = old_f32[vi_tensor]
        sub_new = new_f32[vi_tensor]
        gs = torch.tensor([bp.chain_idx[i] * N + bp.bead_start[i] for i in valid_indices],
                          dtype=torch.long, device=device)
        nm = torch.tensor([bp.n_moved[i] for i in valid_indices],
                          dtype=torch.long, device=device)
        delta_v = self._delta_e_kernel_tiled(
            kernel, pos_f32, sub_old, sub_new, gs, nm,
            V, bp.max_moved, n_total)
        results = torch.zeros(B, dtype=torch.float32, device=device)
        results[vi_tensor] = delta_v
        return results

    def compute_batch_energy_matrices(self, positions_flat, proposals, N):
        is_bp = isinstance(proposals, BatchProposal)
        B = proposals.B if is_bp else len(proposals)
        delta_e = self.compute_batch_delta_energy(positions_flat, proposals, N)
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
                return delta_e, torch.zeros(B, B, dtype=torch.float32, device=device)
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
            return delta_e, torch.zeros(B, B, dtype=torch.float32, device=device)
        r_rep = math.sqrt(r_rep_sq)
        max_moved = old_f32.shape[1]
        nm_tensor = torch.as_tensor(nm_list, dtype=torch.long, device=device)
        move_idx = torch.arange(max_moved, device=device)
        half_mask = move_idx.unsqueeze(0) < nm_tensor.unsqueeze(1)
        valid_mask = torch.cat([half_mask, half_mask], dim=1)
        stacked = torch.cat([old_f32, new_f32], dim=1)
        mask_f = valid_mask.unsqueeze(-1).to(stacked.dtype)
        counts = (2 * nm_tensor).to(stacked.dtype).clamp(min=1).unsqueeze(-1)
        centroids = (stacked * mask_f).sum(dim=1) / counts
        d = stacked - centroids.unsqueeze(1)
        d = d - box * torch.round(d * inv_box)
        dist2 = (d * d).sum(dim=2)
        dist2 = dist2.masked_fill(~valid_mask, 0.0)
        radii = dist2.max(dim=1).values.clamp(min=0).sqrt()
        v_idx = torch.as_tensor(valid_indices, dtype=torch.long, device=device)
        V = v_idx.numel()
        centroids_v = centroids[v_idx]
        radii_v = radii[v_idx]
        dc = centroids_v.unsqueeze(0) - centroids_v.unsqueeze(1)
        dc = dc - box * torch.round(dc * inv_box)
        pair_dist = (dc * dc).sum(dim=2).sqrt()
        rsum = radii_v.unsqueeze(0) + radii_v.unsqueeze(1)
        triu = torch.triu(torch.ones(V, V, dtype=torch.bool, device=device), diagonal=1)
        overlap_mask = (pair_dist < (rsum + r_rep)) & triu
        local_ii, local_jj = torch.nonzero(overlap_mask, as_tuple=True)
        if local_ii.numel() == 0:
            return delta_e, torch.zeros(B, B, dtype=torch.float32, device=device)
        pair_i_t = v_idx[local_ii]
        pair_j_t = v_idx[local_jj]
        emm_kernel = EnergyKernels.emm_correction_kernel()
        correction = emm_kernel(
            old_f32, new_f32, pair_i_t, pair_j_t,
            box, inv_box, r_rep_sq, rep_e, B)
        return delta_e, correction
