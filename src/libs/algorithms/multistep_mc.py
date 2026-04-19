"""MultiStepMC — Migacz rank-1 batched multistep Monte Carlo algorithm."""

import random as _random
import numpy as np
import torch

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.chain.chain_state import ChainState
from rouse_model_python.src.libs.number_space.number_space import NumberSpace
from rouse_model_python.src.libs.energy.energy_computer import EnergyComputer
from rouse_model_python.src.libs.energy.energy_kernels import EnergyKernels
from rouse_model_python.src.libs.mc_moves.batch_proposal import BatchProposal
from rouse_model_python.src.libs.mc_moves.move_proposer import MoveProposer
from rouse_model_python.src.libs.algorithms.metropolis_criterion import MetropolisCriterion
from rouse_model_python.src.libs.algorithms.rand_pool import RandPool
from rouse_model_python.src.libs.algorithms.fused_accept import (
    fused_accept_and_correct,
    fused_accept_and_correct_torch,
)


class MultiStepMC:
    MOVE_SIZE = 100
    REBUILD_INTERVAL = 3
    PIVOT_BATCH_SIZE = 100
    NVTX_ENABLED = False

    @staticmethod
    def _nvtx_push(name: str) -> None:
        if MultiStepMC.NVTX_ENABLED:
            torch.cuda.nvtx.range_push(name)

    @staticmethod
    def _nvtx_pop() -> None:
        if MultiStepMC.NVTX_ENABLED:
            torch.cuda.nvtx.range_pop()

    @staticmethod
    def _drain_pending_stats(pending_stats, stats):
        """Bulk-D2H deferred (move_types, accepted_t, n_moved_np) tuples
        collected during the sweep. One .cpu() call replaces ~20 per-batch
        syncs from the segment phase plus however many from the pivot phase.
        """
        if not pending_stats:
            return
        parts = [t for _, t, _ in pending_stats]
        all_accepted = torch.cat(parts).to(torch.uint8).cpu().numpy()
        offset = 0
        for move_types, accepted_t, n_moved_np in pending_stats:
            B = accepted_t.shape[0]
            slice_acc = all_accepted[offset:offset + B]
            for i in range(B):
                if n_moved_np[i] == 0:
                    continue
                stats.record(move_types[i], bool(slice_acc[i]))
            offset += B

    @staticmethod
    def _process_batch(proposals, state: ChainState, energy_comp: EnergyComputer,
                       positions_flat, gen, cfg: SimulationConfig, stats, ns: NumberSpace,
                       pending_stats=None):
        is_bp = isinstance(proposals, BatchProposal)
        B = proposals.B if is_bp else len(proposals)
        N = cfg.N
        device = cfg.get_torch_device()
        dtype = cfg.dtype
        r_rep_sq = cfg.r_rep_sq
        rep_e = cfg.repulsive_energy

        if cfg.use_batched_mode:
            MultiStepMC._nvtx_push("delta_e_emm")
            delta_e, correction_matrix = \
                energy_comp.compute_batch_energy_matrices(positions_flat, proposals, N)
            MultiStepMC._nvtx_pop()
            if is_bp:
                n_moved_np = np.asarray(proposals.n_moved, dtype=np.int64)
            else:
                n_moved_np = np.array([p.n_moved for p in proposals], dtype=np.int64)
            n_effective = int((n_moved_np > 0).sum())
            if n_effective == 0:
                return
            uniforms_t = torch.rand(n_effective, generator=gen, dtype=dtype, device=device)
            if cfg.accept_on_gpu and is_bp:
                MultiStepMC._nvtx_push("accept_gpu")
                n_moved_t = (proposals.n_moved_t if proposals.n_moved_t is not None
                             else torch.as_tensor(proposals.n_moved, dtype=torch.int64, device=device))
                delta_e_f32 = delta_e if delta_e.dtype == torch.float32 else delta_e.float()
                uniforms_f32 = uniforms_t if uniforms_t.dtype == torch.float32 else uniforms_t.float()
                accepted_t = fused_accept_and_correct_torch(
                    delta_e_f32, correction_matrix, n_moved_t, cfg.kBT, uniforms_f32)
                MultiStepMC._nvtx_pop()
                MultiStepMC._nvtx_push("apply_moves_gpu")
                chain_idx_t = (proposals.chain_idx_t if proposals.chain_idx_t is not None
                               else torch.as_tensor(proposals.chain_idx, dtype=torch.int64, device=device))
                bead_start_t = (proposals.bead_start_t if proposals.bead_start_t is not None
                                else torch.as_tensor(proposals.bead_start, dtype=torch.int64, device=device))
                state.apply_moves_batched(chain_idx_t, bead_start_t,
                                           proposals.new_pos, accepted_t, n_moved_t)
                MultiStepMC._nvtx_pop()
                MultiStepMC._nvtx_push("stats_collect")
                if pending_stats is not None:
                    pending_stats.append((list(proposals.move_types), accepted_t, n_moved_np))
                else:
                    accepted_np = accepted_t.cpu().numpy()
                    for i in range(B):
                        if n_moved_np[i] == 0:
                            continue
                        stats.record(proposals.move_types[i], bool(accepted_np[i]))
                MultiStepMC._nvtx_pop()
            else:
                # Blocking D2H — completes before numpy reads. fused_accept mutates
                # delta_e_np in-place, so copy it out of the CPU tensor view.
                MultiStepMC._nvtx_push("accept_d2h_cpu")
                delta_e_np = delta_e.detach().cpu().numpy().astype(np.float32, copy=True)
                corr_np = correction_matrix.detach().cpu().numpy().astype(np.float32, copy=False)
                uniforms_np = uniforms_t.detach().cpu().numpy().astype(np.float32, copy=False)
                accepted_mask = fused_accept_and_correct(
                    delta_e_np, corr_np, n_moved_np, cfg.kBT, uniforms_np)
                MultiStepMC._nvtx_pop()
                MultiStepMC._nvtx_push("apply_moves")
                for i in range(B):
                    if n_moved_np[i] == 0:
                        continue
                    mtype = proposals.move_types[i] if is_bp else proposals[i].move_type
                    accepted = bool(accepted_mask[i])
                    stats.record(mtype, accepted)
                    if not accepted:
                        continue
                    if is_bp:
                        ci = proposals.chain_idx[i]
                        bs = proposals.bead_start[i]
                        nm_i = int(proposals.n_moved[i])
                        state.apply_move(ci, bs, proposals.new_pos[i, :nm_i], validate=False)
                    else:
                        state.apply_move(proposals[i].chain_idx, proposals[i].bead_start,
                                         proposals[i].new_positions, validate=False)
                MultiStepMC._nvtx_pop()
        else:
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
                accepted = MetropolisCriterion.accept(
                    delta_e[i], cfg.kBT, gen, device, dtype)
                stats.record(proposals[i].move_type, accepted)
                if not accepted:
                    continue
                state.apply_move(proposals[i].chain_idx, proposals[i].bead_start,
                                 proposals[i].new_positions, validate=False)
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
                    e00 = EnergyKernels.compute_segment_pair_energy(
                        orig_i, orig_j, ns, r_rep_sq, rep_e)
                    e10 = EnergyKernels.compute_segment_pair_energy(
                        trial_i, orig_j, ns, r_rep_sq, rep_e)
                    e01 = EnergyKernels.compute_segment_pair_energy(
                        orig_i, trial_j, ns, r_rep_sq, rep_e)
                    e11 = EnergyKernels.compute_segment_pair_energy(
                        trial_i, trial_j, ns, r_rep_sq, rep_e)
                    if e00 == 0.0 and e01 == 0.0 and e10 == 0.0 and e11 == 0.0:
                        continue
                    delta_e[j] += (e11 - e01) - (e10 - e00)

    @staticmethod
    def perform_sweep(state: ChainState, energy_comp: EnergyComputer,
                      gen: torch.Generator, cfg: SimulationConfig, stats,
                      skip_pivot: bool = False):
        seg_info = state.segments
        total_segs = seg_info.total_segments
        N = cfg.N
        ns = state.ns
        device = cfg.get_torch_device()

        batch_size = getattr(cfg, "batch_size", MultiStepMC.MOVE_SIZE)
        segs_per_chain = seg_info.segs_per_chain
        n_chains = cfg.n_chains

        MultiStepMC._nvtx_push("permute_setup")
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
        MultiStepMC._nvtx_pop()

        batched = cfg.use_batched_mode
        pending_stats = [] if (batched and cfg.accept_on_gpu) else None

        # Phase 1: batched segment moves
        positions_flat = state.get_all_flat()
        for b_idx, (b_start, b_end) in enumerate(batch_ranges):
            if not batched and b_idx % MultiStepMC.REBUILD_INTERVAL == 0:
                positions_flat = state.get_all_flat()
                energy_comp.rebuild_cell_list(positions_flat)
            elif batched:
                positions_flat = state.get_all_flat()
            seg_list = []
            for idx in range(b_start, b_end):
                global_seg = perm[idx]
                seg_list.append((seg_info.seg_chain[global_seg],
                                 seg_info.seg_local[global_seg]))
            if batched:
                MultiStepMC._nvtx_push("propose_seg")
                batch_proposals = MoveProposer.propose_batch_segment_moves_fused(
                    state, seg_list, gen, cfg)
                MultiStepMC._nvtx_pop()
            else:
                batch_proposals = [
                    MoveProposer.propose_segment_move(state, ci, ls, gen, cfg)
                    for ci, ls in seg_list
                ]
            MultiStepMC._nvtx_push("process_batch")
            MultiStepMC._process_batch(batch_proposals, state, energy_comp,
                                       positions_flat, gen, cfg, stats, ns,
                                       pending_stats=pending_stats)
            MultiStepMC._nvtx_pop()

        # Phase 2: pivot moves (skipped during production per C4)
        if skip_pivot:
            MultiStepMC._drain_pending_stats(pending_stats, stats)
            return
        if batched:
            MultiStepMC._nvtx_push("pivot_phase")
            chain_perm_t = torch.randperm(cfg.n_chains, generator=gen, device=device)
            if cfg.pivot_on_gpu:
                # GPU path: no CPU mirror, no rand pool, no .tolist() of chain_perm.
                cpu_dev = None
                ns_cpu = None
                positions_cpu = None
                rpool = None
                chain_perm = None
            else:
                cpu_dev = torch.device('cpu')
                ns_cpu = NumberSpace(ns.box_size, ns.sigma, device=cpu_dev, dtype=cfg.dtype)
                positions_cpu = state.positions.detach().cpu()
                rpool = RandPool(gen, device, cfg.dtype, initial_size=cfg.n_chains * 12)
                chain_perm = chain_perm_t.tolist()

            pivot_batch_size = getattr(cfg, "batch_size", MultiStepMC.PIVOT_BATCH_SIZE)
            for pb_start in range(0, cfg.n_chains, pivot_batch_size):
                pb_end = min(pb_start + pivot_batch_size, cfg.n_chains)
                MultiStepMC._nvtx_push("propose_pivot")
                if cfg.pivot_on_gpu:
                    bp = MoveProposer.propose_batch_pivot_moves_gpu(
                        state, chain_perm_t[pb_start:pb_end], gen, cfg)
                else:
                    batch_chains = chain_perm[pb_start:pb_end]
                    bp = MoveProposer.propose_batch_pivot_moves_fused(
                        state, batch_chains, rpool, cfg, ns_cpu, positions_cpu, device)
                MultiStepMC._nvtx_pop()
                positions_flat = state.get_all_flat()
                B = bp.B
                MultiStepMC._nvtx_push("pivot_delta_e_emm")
                delta_e, correction_matrix = \
                    energy_comp.compute_batch_energy_matrices(positions_flat, bp, N)
                MultiStepMC._nvtx_pop()
                n_moved_np = np.asarray(bp.n_moved, dtype=np.int64)
                n_effective = int((n_moved_np > 0).sum())
                if n_effective == 0:
                    continue
                uniforms_t = torch.rand(n_effective, generator=gen, dtype=cfg.dtype, device=device)
                if cfg.accept_on_gpu:
                    MultiStepMC._nvtx_push("pivot_accept_gpu")
                    n_moved_t = (bp.n_moved_t if bp.n_moved_t is not None
                                 else torch.as_tensor(bp.n_moved, dtype=torch.int64, device=device))
                    delta_e_f32 = delta_e if delta_e.dtype == torch.float32 else delta_e.float()
                    uniforms_f32 = uniforms_t if uniforms_t.dtype == torch.float32 else uniforms_t.float()
                    accepted_t = fused_accept_and_correct_torch(
                        delta_e_f32, correction_matrix, n_moved_t, cfg.kBT, uniforms_f32)
                    MultiStepMC._nvtx_pop()
                    MultiStepMC._nvtx_push("pivot_apply_gpu")
                    chain_idx_t = (bp.chain_idx_t if bp.chain_idx_t is not None
                                   else torch.as_tensor(bp.chain_idx, dtype=torch.int64, device=device))
                    bead_start_t = (bp.bead_start_t if bp.bead_start_t is not None
                                    else torch.as_tensor(bp.bead_start, dtype=torch.int64, device=device))
                    state.apply_moves_batched(chain_idx_t, bead_start_t,
                                               bp.new_pos, accepted_t, n_moved_t)
                    MultiStepMC._nvtx_pop()
                    MultiStepMC._nvtx_push("pivot_stats_collect")
                    if pending_stats is not None:
                        pending_stats.append((['pivot'] * B, accepted_t, n_moved_np))
                    else:
                        accepted_np = accepted_t.cpu().numpy()
                        for i in range(B):
                            if n_moved_np[i] == 0:
                                continue
                            stats.record('pivot', bool(accepted_np[i]))
                    MultiStepMC._nvtx_pop()
                else:
                    MultiStepMC._nvtx_push("pivot_accept_d2h_cpu")
                    delta_e_np = delta_e.detach().cpu().numpy().astype(np.float32, copy=True)
                    corr_np = correction_matrix.detach().cpu().numpy().astype(np.float32, copy=False)
                    uniforms_np = uniforms_t.detach().cpu().numpy().astype(np.float32, copy=False)
                    accepted_mask = fused_accept_and_correct(
                        delta_e_np, corr_np, n_moved_np, cfg.kBT, uniforms_np)
                    MultiStepMC._nvtx_pop()
                    MultiStepMC._nvtx_push("pivot_apply_moves")
                    for i in range(B):
                        if n_moved_np[i] == 0:
                            continue
                        accepted = bool(accepted_mask[i])
                        stats.record('pivot', accepted)
                        if not accepted:
                            continue
                        ci = bp.chain_idx[i]
                        bs = bp.bead_start[i]
                        nm = int(bp.n_moved[i])
                        new_pos_gpu = bp.new_pos[i, :nm]
                        state.apply_move(ci, bs, new_pos_gpu, validate=False)
                    MultiStepMC._nvtx_pop()
            if positions_cpu is not None:
                # Legacy path only — under cfg.pivot_on_gpu we never built it.
                positions_cpu.copy_(state.positions.detach().cpu())
            MultiStepMC._nvtx_pop()
        else:
            positions_flat = state.get_all_flat()
            energy_comp.rebuild_cell_list(positions_flat)
            for c in range(cfg.n_chains):
                proposal = MoveProposer.propose_pivot_move(state, c, gen, cfg)
                if proposal.n_moved == 0:
                    continue
                delta_e = energy_comp.compute_delta_energy_move(
                    positions_flat, proposal.chain_idx, proposal.bead_start,
                    proposal.old_positions, proposal.new_positions, N)
                accepted = MetropolisCriterion.accept(
                    delta_e, cfg.kBT, gen, device, cfg.dtype)
                stats.record('pivot', accepted)
                if accepted:
                    state.apply_move(proposal.chain_idx, proposal.bead_start,
                                     proposal.new_positions, validate=False)

        MultiStepMC._drain_pending_stats(pending_stats, stats)
