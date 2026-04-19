"""ConventionalMC — classical sequential single-trial Metropolis baseline.

Default: per-proposal CPU cell-list rebuild via EnergyComputer.compute_delta_energy_move.
When cfg.use_gpu_energy_path is True: route the per-proposal delta-E through the
shared GPU-native fused kernel (EnergyComputer.compute_batch_delta_energy with B=1),
skipping the cell-list rebuild entirely. Same EnergyComputer instance, same cfg —
fairness anchor 6.1 preserved.
"""

import random as _random
import torch

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.chain.chain_state import ChainState
from rouse_model_python.src.libs.energy.energy_computer import EnergyComputer
from rouse_model_python.src.libs.mc_moves.move_proposer import MoveProposer
from rouse_model_python.src.libs.mc_moves.batch_proposal import BatchProposal
from rouse_model_python.src.libs.algorithms.metropolis_criterion import MetropolisCriterion


class ConventionalMC:
    @staticmethod
    def _delta_e_gpu_single(energy_comp: EnergyComputer, positions_flat,
                            proposal, N: int, scratch_bp: BatchProposal) -> float:
        nm = proposal.n_moved
        scratch_bp.old_pos[0, :nm] = proposal.old_positions
        scratch_bp.new_pos[0, :nm] = proposal.new_positions
        scratch_bp.chain_idx[0] = proposal.chain_idx
        scratch_bp.bead_start[0] = proposal.bead_start
        scratch_bp.n_moved[0] = nm
        scratch_bp.move_types[0] = proposal.move_type
        scratch_bp.valid[0] = True
        scratch_bp._old_f32 = None
        scratch_bp._new_f32 = None
        delta_v = energy_comp.compute_batch_delta_energy(positions_flat, scratch_bp, N)
        return float(delta_v[0].item())

    @staticmethod
    def run_sweep(state: ChainState, energy_comp: EnergyComputer,
                  gen: torch.Generator, cfg: SimulationConfig, stats,
                  skip_pivot: bool = False):
        seg_info = state.segments
        total_segs = seg_info.total_segments
        N = cfg.N
        device = cfg.get_torch_device()
        dtype = cfg.dtype
        use_gpu_energy = bool(getattr(cfg, "use_gpu_energy_path", False))

        rng_py = _random.Random(gen.initial_seed())
        perm = list(range(total_segs))
        rng_py.shuffle(perm)

        scratch_bp = None
        if use_gpu_energy:
            max_moved = max(cfg.residues_per_segment, N)
            scratch_bp = BatchProposal(B=1, max_moved=max_moved, device=device, dtype=dtype)

        # ── Phase 1: sequential segment moves ────────────────────────────
        for count, global_seg in enumerate(perm):
            ci = seg_info.seg_chain[global_seg]
            ls = seg_info.seg_local[global_seg]
            positions_flat = state.get_all_flat()

            if not use_gpu_energy:
                energy_comp.rebuild_cell_list(positions_flat)

            proposal = MoveProposer.propose_segment_move(state, ci, ls, gen, cfg)
            if proposal.n_moved == 0:
                continue

            if use_gpu_energy:
                delta_e = ConventionalMC._delta_e_gpu_single(
                    energy_comp, positions_flat, proposal, N, scratch_bp)
            else:
                delta_e = energy_comp.compute_delta_energy_move(
                    positions_flat, proposal.chain_idx, proposal.bead_start,
                    proposal.old_positions, proposal.new_positions, N,
                )

            accepted = MetropolisCriterion.accept(delta_e, cfg.kBT, gen, device, dtype)
            stats.record(proposal.move_type, accepted)
            if accepted:
                state.apply_move(proposal.chain_idx, proposal.bead_start,
                                 proposal.new_positions, validate=False)

        # ── Phase 2: sequential pivot moves (skipped during production) ──
        if skip_pivot:
            return
        for c in range(cfg.n_chains):
            positions_flat = state.get_all_flat()

            if not use_gpu_energy:
                energy_comp.rebuild_cell_list(positions_flat)

            proposal = MoveProposer.propose_pivot_move(state, c, gen, cfg)
            if proposal.n_moved == 0:
                continue

            if use_gpu_energy:
                delta_e = ConventionalMC._delta_e_gpu_single(
                    energy_comp, positions_flat, proposal, N, scratch_bp)
            else:
                delta_e = energy_comp.compute_delta_energy_move(
                    positions_flat, proposal.chain_idx, proposal.bead_start,
                    proposal.old_positions, proposal.new_positions, N,
                )
            accepted = MetropolisCriterion.accept(delta_e, cfg.kBT, gen, device, dtype)
            stats.record('pivot', accepted)
            if accepted:
                state.apply_move(proposal.chain_idx, proposal.bead_start,
                                 proposal.new_positions, validate=False)
