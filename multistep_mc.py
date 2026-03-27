"""
Matrix-based segmented multistep Monte Carlo algorithm.
GPU optimization: pre-generates random numbers in bulk to eliminate
per-call GPU→CPU synchronization (~0.15ms × thousands of calls).

Implements the batched multistep MC scheme from the C# SegmentedMultiStepMC:

  1. Segments are shuffled and grouped into batches of size MOVE_SIZE.
  2. Within each batch, ALL moves are proposed from the batch-start
     configuration before any acceptance decision.
  3. Initial ΔE for each proposal is computed via the cell-list energy computer.
  4. Acceptance proceeds sequentially within the batch using the standard
     Metropolis criterion. When move k is accepted, the energy estimates
     for remaining moves j > k are corrected via a rank-1 update derived
     from the four pairwise segment energy matrices:

       Emm00[k,j] = pair_e(orig_k, orig_j)
       Emm01[k,j] = pair_e(orig_k, trial_j)
       Emm10[k,j] = pair_e(trial_k, orig_j)
       Emm11[k,j] = pair_e(trial_k, trial_j)

     Correction to ΔE[j]:
       ΔE[j] += (Emm11[k,j] - Emm01[k,j]) - (Emm10[k,j] - Emm00[k,j])

  5. The cell-list is rebuilt between batches to reflect accepted moves.
  6. After all segment batches: one pivot move per chain (sequential
     Metropolis with cell-list ΔE).

Reference:
  C# source: SegmentedMultiStepMC.cs, SegmentedEnergyMatrices.cs
"""

import math
import torch

from .config import SimulationConfig
from .chain import ChainState
from .energy import EnergyComputer, compute_segment_pair_energy
from .mc_moves import (propose_segment_move, propose_pivot_move,
                       propose_pivot_move_pooled,
                       propose_batch_segment_moves, MoveProposal)
from .number_space import NumberSpace


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOVE_SIZE = 10  # Batch size for multistep MC


class RandPool:
    """Pre-generated random number pool to avoid per-call GPU→CPU sync.

    One bulk torch.rand() + .tolist() replaces thousands of individual
    torch.rand(1).item() calls, each of which forces a GPU→CPU sync
    (~0.15ms latency).
    """

    def __init__(self, gen: torch.Generator, device: torch.device,
                 dtype: torch.dtype, initial_size: int = 10000):
        self._gen = gen
        self._device = device
        self._dtype = dtype
        self._pool = torch.rand(
            initial_size, generator=gen, dtype=dtype, device=device).tolist()
        self._idx = 0

    def next(self) -> float:
        """Return next pre-generated random float in [0, 1)."""
        if self._idx >= len(self._pool):
            more = torch.rand(
                5000, generator=self._gen, dtype=self._dtype,
                device=self._device).tolist()
            self._pool.extend(more)
        val = self._pool[self._idx]
        self._idx += 1
        return val


# ---------------------------------------------------------------------------
# Metropolis acceptance
# ---------------------------------------------------------------------------

def metropolis_accept(delta_e: float, kBT: float, gen: torch.Generator,
                       device: torch.device, dtype: torch.dtype,
                       rand_pool: RandPool = None) -> bool:
    """Standard Metropolis acceptance criterion with overflow guards."""
    if delta_e <= 0.0:
        return True
    exponent = -delta_e / kBT
    if exponent <= -745.0:
        return False
    if exponent >= 709.0:
        return True
    prob = math.exp(exponent)
    if rand_pool is not None:
        r = rand_pool.next()
    else:
        r = torch.rand(1, generator=gen, dtype=dtype, device=device).item()
    return r < prob


# ---------------------------------------------------------------------------
# Rank-1 energy matrix update within a batch
# ---------------------------------------------------------------------------

def _process_batch(proposals, state, energy_comp, positions_flat,
                   gen, cfg, stats, ns):
    """
    Process one batch of segment move proposals with the multistep MC
    algorithm (propose all → sequential accept with rank-1 updates).

    Batched mode: pre-computes E_total + full E_mm[i,j] energy matrices
    on GPU in parallel. Sequential acceptance reads pre-computed values.

    Sequential mode: computes delta-E individually with cell-list, then
    rank-1 corrections via cell-proximity filtering.
    """
    B = len(proposals)
    N = cfg.N
    device = cfg.get_torch_device()
    dtype = cfg.dtype
    r_rep_sq = cfg.r_rep_sq
    rep_e = cfg.repulsive_energy

    if cfg.use_batched_mode:
        # ── BATCHED PATH: parallel energy matrix computation ──────────
        # Pre-compute ALL energies in parallel (paper's core optimization).
        # E_total[i]: proposal i vs stationary system
        # E_mm{00,01,10,11}[i,j]: pairwise segment energies for rank-1
        delta_e, Emm00, Emm01, Emm10, Emm11 = \
            energy_comp.compute_batch_energy_matrices(
                positions_flat, proposals, N)

        # RandPool: one bulk GPU transfer instead of B individual syncs
        rpool = RandPool(gen, device, dtype, initial_size=B * 2)

        # Sequential acceptance — just reads pre-computed matrix elements
        for i in range(B):
            if proposals[i].n_moved == 0:
                continue

            accepted = metropolis_accept(delta_e[i], cfg.kBT, gen, device,
                                         dtype, rand_pool=rpool)
            stats.record(proposals[i].move_type, accepted)

            if not accepted:
                continue

            state.apply_move(proposals[i].chain_idx, proposals[i].bead_start,
                             proposals[i].new_positions)

            # Rank-1 update: pure scalar reads from pre-computed matrices
            for j in range(i + 1, B):
                if proposals[j].n_moved == 0:
                    continue
                e00 = Emm00[i, j].item()
                e01 = Emm01[i, j].item()
                e10 = Emm10[i, j].item()
                e11 = Emm11[i, j].item()
                correction = (e11 - e01) - (e10 - e00)
                if correction != 0.0:
                    delta_e[j] += correction

    else:
        # ── SEQUENTIAL PATH: cell-list based ──────────────────────────
        # Precompute cell-neighbor sets for proximity filtering
        cl = energy_comp.cell_list
        nc = cl._nc; inv_cs = cl._inv_cs
        half_box = cl._half_box; nc_m1 = nc - 1
        neighbor_cells_table = cl._neighbor_cells

        proposals_lists = []
        for p in proposals:
            if p.n_moved > 0:
                proposals_lists.append(
                    (p.old_positions.tolist(), p.new_positions.tolist()))
            else:
                proposals_lists.append(None)

        cell_nbr_sets = []
        for pi in range(B):
            if proposals_lists[pi] is None:
                cell_nbr_sets.append(None)
                continue
            beads_old, beads_new = proposals_lists[pi]
            cells = set()
            for bead in beads_old + beads_new:
                cx = max(0, min(nc_m1, int((bead[0] + half_box) * inv_cs)))
                cy = max(0, min(nc_m1, int((bead[1] + half_box) * inv_cs)))
                cz = max(0, min(nc_m1, int((bead[2] + half_box) * inv_cs)))
                cells.add((cx * nc + cy) * nc + cz)
            expanded = set()
            for c in cells:
                expanded.update(neighbor_cells_table[c])
            cell_nbr_sets.append(expanded)

        delta_e = []
        for p in proposals:
            if p.n_moved == 0:
                delta_e.append(0.0)
                continue
            de = energy_comp.compute_delta_energy_move(
                positions_flat, p.chain_idx, p.bead_start,
                p.old_positions, p.new_positions, N)
            delta_e.append(de)

        for i in range(B):
            if proposals[i].n_moved == 0:
                continue

            accepted = metropolis_accept(delta_e[i], cfg.kBT, gen, device, dtype)
            stats.record(proposals[i].move_type, accepted)

            if not accepted:
                continue

            state.apply_move(proposals[i].chain_idx, proposals[i].bead_start,
                             proposals[i].new_positions)

            orig_i = proposals[i].old_positions
            trial_i = proposals[i].new_positions
            nbr_i = cell_nbr_sets[i]

            for j in range(i + 1, B):
                if proposals[j].n_moved == 0:
                    continue
                nbr_j = cell_nbr_sets[j]
                if nbr_i is not None and nbr_j is not None and nbr_i.isdisjoint(nbr_j):
                    continue
                orig_j = proposals[j].old_positions
                trial_j = proposals[j].new_positions
                e00 = compute_segment_pair_energy(orig_i, orig_j, ns, r_rep_sq, rep_e)
                e10 = compute_segment_pair_energy(trial_i, orig_j, ns, r_rep_sq, rep_e)
                e01 = compute_segment_pair_energy(orig_i, trial_j, ns, r_rep_sq, rep_e)
                e11 = compute_segment_pair_energy(trial_i, trial_j, ns, r_rep_sq, rep_e)
                if e00 == 0.0 and e01 == 0.0 and e10 == 0.0 and e11 == 0.0:
                    continue
                delta_e[j] += (e11 - e01) - (e10 - e00)


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def perform_sweep(state: ChainState, energy_comp: EnergyComputer,
                   gen: torch.Generator, cfg: SimulationConfig,
                   stats):
    """
    Perform one full MC sweep (segment moves + pivot moves).

    Phase 1 — Batched multistep segment moves:
      All segments are randomly permuted, then processed in batches of
      MOVE_SIZE. The cell-list is rebuilt at the start of each batch.

    Phase 2 — Sequential pivot moves:
      One pivot move proposed per chain with immediate accept/reject.
      Cell-list rebuilt once before the loop; positions_flat is a view
      that auto-updates when state.apply_move() modifies state.positions.
    """
    seg_info = state.segments
    total_segs = seg_info.total_segments
    N = cfg.N
    ns = state.ns
    device = cfg.get_torch_device()

    # Random permutation of all segments
    perm = torch.randperm(total_segs, generator=gen, device=device)

    # ── Phase 1: Batched multistep segment moves ─────────────────────
    batch_size = MOVE_SIZE
    n_batches = (total_segs + batch_size - 1) // batch_size
    # Rebuild cell list every REBUILD_INTERVAL batches to amortize build cost.
    # Between rebuilds, the cell list is slightly stale but the rank-1
    # corrections and auto-updating positions_flat view keep energy accurate.
    REBUILD_INTERVAL = 3

    batched = cfg.use_batched_mode

    for b in range(n_batches):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, total_segs)

        # Rebuild cell-list periodically (sequential mode needs it for
        # delta-E and rank-1; batched mode skips — uses direct pairwise)
        if not batched and b % REBUILD_INTERVAL == 0:
            positions_flat = state.get_all_flat()
            energy_comp.rebuild_cell_list(positions_flat)
        elif batched:
            positions_flat = state.get_all_flat()

        # Propose all moves in this batch
        seg_list = []
        for idx in range(b_start, b_end):
            global_seg = perm[idx].item()
            seg_list.append((seg_info.seg_chain[global_seg],
                             seg_info.seg_local[global_seg]))

        if batched:
            batch_proposals = propose_batch_segment_moves(
                state, seg_list, gen, cfg)
        else:
            batch_proposals = [
                propose_segment_move(state, ci, ls, gen, cfg)
                for ci, ls in seg_list
            ]

        # Multistep accept/reject with rank-1 updates
        _process_batch(batch_proposals, state, energy_comp, positions_flat,
                       gen, cfg, stats, ns)

    # ── Phase 2: Pivot moves (one per chain, sequential) ──────────────
    # For GPU: run entire pivot loop on CPU to avoid thousands of GPU→CPU
    # syncs. One bulk transfer at start + end instead of ~3000 syncs.
    if device.type != 'cpu':
        from .number_space import NumberSpace as NS
        ns_cpu = NS(ns.box_size, ns.sigma, device=torch.device('cpu'), dtype=cfg.dtype)

        # Transfer positions to CPU once
        positions_cpu = state.positions.detach().cpu()  # [n_chains, N, 3]
        pf_cpu = positions_cpu.reshape(-1, 3)
        energy_comp.rebuild_cell_list(pf_cpu)

        # Pre-generate ALL random numbers for pivots (1 GPU transfer)
        rpool = RandPool(gen, device, cfg.dtype,
                         initial_size=cfg.n_chains * 12)

        for c in range(cfg.n_chains):
            proposal = propose_pivot_move_pooled(
                positions_cpu, c, rpool, ns_cpu, cfg, device_out=torch.device('cpu'))

            if proposal.n_moved == 0:
                continue

            delta_e = energy_comp.compute_delta_energy_move(
                pf_cpu, proposal.chain_idx, proposal.bead_start,
                proposal.old_positions, proposal.new_positions, N
            )

            accepted = metropolis_accept(delta_e, cfg.kBT, gen, device,
                                         cfg.dtype, rand_pool=rpool)
            stats.record('pivot', accepted)

            if accepted:
                bs = proposal.bead_start
                nm = proposal.n_moved
                positions_cpu[c, bs:bs + nm] = proposal.new_positions
                pf_cpu = positions_cpu.reshape(-1, 3)

        # Transfer final positions back to GPU
        state.positions.copy_(positions_cpu.to(device))
    else:
        positions_flat = state.get_all_flat()
        energy_comp.rebuild_cell_list(positions_flat)

        for c in range(cfg.n_chains):
            proposal = propose_pivot_move(state, c, gen, cfg)

            if proposal.n_moved == 0:
                continue

            delta_e = energy_comp.compute_delta_energy_move(
                positions_flat, proposal.chain_idx, proposal.bead_start,
                proposal.old_positions, proposal.new_positions, N
            )

            accepted = metropolis_accept(delta_e, cfg.kBT, gen, device, cfg.dtype)
            stats.record('pivot', accepted)

            if accepted:
                state.apply_move(proposal.chain_idx, proposal.bead_start,
                                 proposal.new_positions)
