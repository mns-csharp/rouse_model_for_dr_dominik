"""BruteforceValidator — reference all-pairs delta-E for validation."""

import torch

from rouse_model_python.src.libs.number_space.number_space import NumberSpace
from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.chain.chain_state import ChainState
from rouse_model_python.src.libs.mc_moves.move_proposer import MoveProposer
from rouse_model_python.src.libs.energy.energy_computer import EnergyComputer


class BruteforceValidator:
    @staticmethod
    def compute_delta_energy_bruteforce(positions_flat, chain_idx, bead_start,
                                        old_positions, new_positions,
                                        N, ns: NumberSpace,
                                        r_rep_sq: float,
                                        repulsive_energy: float) -> float:
        n_moved = old_positions.shape[0]
        n_total = positions_flat.shape[0]
        global_start = chain_idx * N + bead_start
        global_end = global_start + n_moved
        box = ns.box_size
        inv_box = ns._inv_box
        mask = torch.ones(n_total, dtype=torch.bool, device=positions_flat.device)
        mask[global_start:global_end] = False
        stationary = positions_flat[mask]
        delta_e = 0.0
        if stationary.shape[0] > 0:
            d_old = stationary.unsqueeze(0) - old_positions.unsqueeze(1)
            d_old = d_old - box * torch.round(d_old * inv_box)
            r2_old = (d_old * d_old).sum(dim=2)
            e_old = (r2_old < r_rep_sq).sum().item()
            d_new = stationary.unsqueeze(0) - new_positions.unsqueeze(1)
            d_new = d_new - box * torch.round(d_new * inv_box)
            r2_new = (d_new * d_new).sum(dim=2)
            e_new = (r2_new < r_rep_sq).sum().item()
            delta_e += (e_new - e_old) * repulsive_energy
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

    @staticmethod
    def validate_cell_list_vs_bruteforce(energy_comp: EnergyComputer,
                                         state: ChainState,
                                         cfg: SimulationConfig,
                                         n_samples: int = 10) -> dict:
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
            proposal = MoveProposer.propose_segment_move(state, chain_idx, local_seg, gen, cfg)
            if proposal.n_moved == 0:
                continue
            de_cell = energy_comp.compute_delta_energy_move(
                positions_flat, proposal.chain_idx, proposal.bead_start,
                proposal.old_positions, proposal.new_positions, N)
            de_brute = BruteforceValidator.compute_delta_energy_bruteforce(
                positions_flat, proposal.chain_idx, proposal.bead_start,
                proposal.old_positions, proposal.new_positions,
                N, ns, cfg.r_rep_sq, cfg.repulsive_energy)
            err = abs(de_cell - de_brute)
            max_err = max(max_err, err)
            details.append({'cell_list': de_cell, 'bruteforce': de_brute, 'error': err})
        return {'max_abs_error': max_err, 'all_match': max_err == 0.0, 'details': details}
