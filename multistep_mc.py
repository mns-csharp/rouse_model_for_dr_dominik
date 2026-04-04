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
                       propose_batch_pivot_moves,
                       propose_batch_segment_moves,
                       propose_batch_segment_moves_fused, MoveProposal)
from .number_space import NumberSpace


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOVE_SIZE = 20  # Batch size for multistep MC (larger = fewer batches = less overhead)


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
    from .batch_proposal import BatchProposal
    is_bp = isinstance(proposals, BatchProposal)
    B = proposals.B if is_bp else len(proposals)
    N = cfg.N
    device = cfg.get_torch_device()
    dtype = cfg.dtype
    r_rep_sq = cfg.r_rep_sq
    rep_e = cfg.repulsive_energy

    if cfg.use_batched_mode:
        from .batch_proposal import BatchProposal

        # ── BATCHED PATH: parallel energy matrix computation ──────────
        # Pre-compute ALL energies in parallel (Migacz et al.).
        # E_total[i]: proposal i vs stationary system
        # correction[i,j]: fused rank-1 correction = (E11-E01)-(E10-E00)
        delta_e, correction_matrix = \
            energy_comp.compute_batch_energy_matrices(
                positions_flat, proposals, N)

        # RandPool: one bulk GPU transfer instead of B individual syncs
        rpool = RandPool(gen, device, dtype, initial_size=B * 2)

        is_bp = isinstance(proposals, BatchProposal)

        # Pre-fetch correction as Python floats to avoid per-element .item()
        corr = correction_matrix.tolist()

        # Sequential acceptance — reads pre-computed values only
        for i in range(B):
            nm_i = proposals.n_moved[i] if is_bp else proposals[i].n_moved
            if nm_i == 0:
                continue

            mtype = proposals.move_types[i] if is_bp else proposals[i].move_type

            accepted = metropolis_accept(delta_e[i], cfg.kBT, gen, device,
                                         dtype, rand_pool=rpool)
            stats.record(mtype, accepted)

            if not accepted:
                continue

            if is_bp:
                ci = proposals.chain_idx[i]
                bs = proposals.bead_start[i]
                state.apply_move(ci, bs, proposals.new_pos[i, :nm_i])
            else:
                state.apply_move(proposals[i].chain_idx, proposals[i].bead_start,
                                 proposals[i].new_positions)

            # Rank-1 update from pre-computed fused correction matrix
            for j in range(i + 1, B):
                nm_j = proposals.n_moved[j] if is_bp else proposals[j].n_moved
                if nm_j == 0:
                    continue
                c = corr[i][j]
                if c != 0.0:
                    delta_e[j] += c

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

    # ── Build batched permutation with same-chain exclusion ──────────
    # Group segments into rounds (one segment per chain per round) so
    # that within each batch, all segments are from different chains.
    # This prevents the multistep algorithm from breaking boundary bonds
    # when two same-chain segments are accepted from the same pre-batch state.
    import random as _random

    batch_size = MOVE_SIZE
    segs_per_chain = seg_info.segs_per_chain
    n_chains = cfg.n_chains

    buckets = [[] for _ in range(n_chains)]
    for gs in range(total_segs):
        buckets[seg_info.seg_chain[gs]].append(gs)
    rng_py = _random.Random(gen.initial_seed())
    for bucket in buckets:
        rng_py.shuffle(bucket)

    rounds = []
    for k in range(segs_per_chain):
        round_segs = [bucket[k] for bucket in buckets if k < len(bucket)]
        rng_py.shuffle(round_segs)
        rounds.append(round_segs)

    perm = []
    round_boundaries = []
    for round_segs in rounds:
        round_boundaries.append(len(perm))
        perm.extend(round_segs)
    round_boundaries.append(len(perm))

    batch_ranges = []
    for ri in range(len(rounds)):
        r_start = round_boundaries[ri]
        r_end = round_boundaries[ri + 1]
        for bs in range(r_start, r_end, batch_size):
            batch_ranges.append((bs, min(bs + batch_size, r_end)))

    # Rebuild cell list every REBUILD_INTERVAL batches to amortize build cost.
    REBUILD_INTERVAL = 3

    batched = cfg.use_batched_mode

    # ── Phase 1: Batched multistep segment moves ─────────────────────
    for b_idx, (b_start, b_end) in enumerate(batch_ranges):

        # Rebuild cell-list periodically (sequential mode needs it for
        # delta-E and rank-1; batched mode skips — uses direct pairwise)
        if not batched and b_idx % REBUILD_INTERVAL == 0:
            positions_flat = state.get_all_flat()
            energy_comp.rebuild_cell_list(positions_flat)
        elif batched:
            positions_flat = state.get_all_flat()

        # Propose all moves in this batch
        seg_list = []
        for idx in range(b_start, b_end):
            global_seg = perm[idx]
            seg_list.append((seg_info.seg_chain[global_seg],
                             seg_info.seg_local[global_seg]))

        if batched:
            batch_proposals = propose_batch_segment_moves_fused(
                state, seg_list, gen, cfg)
        else:
            batch_proposals = [
                propose_segment_move(state, ci, ls, gen, cfg)
                for ci, ls in seg_list
            ]

        # Multistep accept/reject with rank-1 updates
        _process_batch(batch_proposals, state, energy_comp, positions_flat,
                       gen, cfg, stats, ns)

    # ── Phase 2: Pivot moves ────────────────────────────────────────────
    # Batched mode: propose pivots in batches of PIVOT_BATCH_SIZE, use
    # matrix-based energy computation + rank-1 corrections (same as
    # segment moves). This eliminates hundreds of sequential delta_energy
    # calls that dominated the pivot phase.
    #
    # Sequential mode: one pivot per chain with cell-list delta-E.

    PIVOT_BATCH_SIZE = MOVE_SIZE  # same batch size as segment moves

    if batched:
        from .number_space import NumberSpace as NS
        from .mc_moves import propose_batch_pivot_moves_fused

        # Work on CPU to avoid GPU→CPU syncs during unwrapping
        cpu_dev = torch.device('cpu')
        ns_cpu = NS(ns.box_size, ns.sigma, device=cpu_dev, dtype=cfg.dtype)

        positions_cpu = state.positions.detach().cpu()  # [n_chains, N, 3]

        # Pre-generate random numbers for all pivots
        rpool = RandPool(gen, device, cfg.dtype,
                         initial_size=cfg.n_chains * 12)

        # Shuffle chain order for pivot proposals
        chain_perm = torch.randperm(cfg.n_chains, generator=gen, device=device).tolist()

        # Process pivots in batches with matrix-based energy on GPU
        for pb_start in range(0, cfg.n_chains, PIVOT_BATCH_SIZE):
            pb_end = min(pb_start + PIVOT_BATCH_SIZE, cfg.n_chains)
            batch_chains = chain_perm[pb_start:pb_end]

            # Propose all pivots (CPU) and pack into GPU BatchProposal
            bp = propose_batch_pivot_moves_fused(
                state, batch_chains, rpool, cfg, ns_cpu, positions_cpu, device)

            # Energy matrices computed on GPU via BatchProposal (no repack)
            positions_flat = state.get_all_flat()
            B = bp.B
            delta_e, correction_matrix = \
                energy_comp.compute_batch_energy_matrices(
                    positions_flat, bp, N)

            # Pre-fetch correction as Python floats
            corr = correction_matrix.tolist()

            # Sequential acceptance with rank-1 corrections
            for i in range(B):
                if bp.n_moved[i] == 0:
                    continue

                accepted = metropolis_accept(delta_e[i], cfg.kBT, gen, device,
                                             cfg.dtype, rand_pool=rpool)
                stats.record('pivot', accepted)

                if not accepted:
                    continue

                # Apply to both CPU and GPU positions
                ci = bp.chain_idx[i]
                bs = bp.bead_start[i]
                nm = bp.n_moved[i]
                new_pos_gpu = bp.new_pos[i, :nm]
                positions_cpu[ci, bs:bs + nm] = new_pos_gpu.cpu()
                state.positions[ci, bs:bs + nm] = new_pos_gpu
                state._validate_boundary_bonds(ci, bs, nm)

                # Rank-1 update from pre-computed fused correction matrix
                for j in range(i + 1, B):
                    if bp.n_moved[j] == 0:
                        continue
                    c = corr[i][j]
                    if c != 0.0:
                        delta_e[j] += c

        # Ensure GPU state is up to date
        state.positions.copy_(positions_cpu.to(device))

    else:
        # Sequential mode: cell-list based pivot moves
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
