"""FastSweep — full MC sweep orchestrator for the numpy/numba fast backend.

Mirrors multistep_mc.perform_sweep() but operates on numpy positions with
Numba-JIT'd delta-E and scalar Rodrigues rotations.

Phase 1: batched segment moves with rank-1 corrections (Migacz et al.)
Phase 2: sequential pivot moves with cell-list delta-E
"""

from rouse_model_python.src.libs.chain.segment_info import SegmentInfo
from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.paths.fast.fast_energy_computer import FastEnergyComputer
from rouse_model_python.src.libs.paths.fast.fast_kernels import FastKernels
from rouse_model_python.src.libs.paths.fast.fast_proposer import FastProposer
from rouse_model_python.src.libs.simulation.simulation_stats import SimulationStats


class FastSweep:
    MOVE_SIZE = 20
    REBUILD_INTERVAL = 3

    @staticmethod
    def perform_pivot_phase(positions_np, seg_info: SegmentInfo,
                            energy_comp: FastEnergyComputer,
                            cfg: SimulationConfig, stats: SimulationStats,
                            rng) -> None:
        N = cfg.N
        n_chains = cfg.n_chains
        box = cfg.box_size
        inv_box = 1.0 / box
        half_box = box / 2.0
        kBT = cfg.kBT

        pos_flat = positions_np.reshape(-1, 3)
        energy_comp.rebuild_cell_list(pos_flat)

        chain_perm = rng.permutation(n_chains)
        for c_idx in chain_perm:
            c = int(c_idx)
            chain_pos = positions_np[c]
            p = FastProposer.propose_pivot(chain_pos, c, N, box, inv_box, half_box, rng, cfg.l0)
            if p.n_moved == 0:
                continue
            de = energy_comp.compute_delta_energy(
                p.chain_idx, p.bead_start, p.n_moved,
                p.old_pos, p.new_pos, N)
            accepted = FastProposer.metropolis_accept(de, kBT, rng)
            stats.record('pivot', accepted)
            if accepted:
                positions_np[p.chain_idx, p.bead_start:p.bead_start + p.n_moved] = p.new_pos
                FastProposer.validate_boundary_bonds(
                    positions_np, p.chain_idx, p.bead_start,
                    p.n_moved, N, cfg.l0, box, inv_box)

    @staticmethod
    def perform_sweep(positions_np, seg_info: SegmentInfo,
                      energy_comp: FastEnergyComputer,
                      cfg: SimulationConfig, stats: SimulationStats,
                      rng, skip_pivot: bool = False) -> None:
        N = cfg.N
        n_chains = cfg.n_chains
        total_segs = seg_info.total_segments
        box = cfg.box_size
        inv_box = 1.0 / box
        half_box = box / 2.0
        kBT = cfg.kBT
        max_angle = cfg.max_angle_hinge

        batch_size = FastSweep.MOVE_SIZE
        segs_per_chain = seg_info.segs_per_chain

        buckets = [[] for _ in range(n_chains)]
        for gs in range(total_segs):
            buckets[seg_info.seg_chain[gs]].append(gs)
        for bucket in buckets:
            rng.shuffle(bucket)

        rounds = []
        for k in range(segs_per_chain):
            round_segs = [bucket[k] for bucket in buckets if k < len(bucket)]
            rng.shuffle(round_segs)
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
                be = min(bs + batch_size, r_end)
                batch_ranges.append((bs, be))

        for b_idx, (b_start, b_end) in enumerate(batch_ranges):

            if b_idx % FastSweep.REBUILD_INTERVAL == 0:
                pos_flat = positions_np.reshape(-1, 3)
                energy_comp.rebuild_cell_list(pos_flat)

            proposals = []
            for idx in range(b_start, b_end):
                gs = perm[idx]
                chain_idx = seg_info.seg_chain[gs]
                local_seg = seg_info.seg_local[gs]
                chain_pos = positions_np[chain_idx]
                p = FastProposer.propose_segment_move(
                    chain_pos, chain_idx, local_seg, seg_info, N,
                    box, inv_box, half_box, max_angle, rng, cfg.l0)
                proposals.append(p)

            B = len(proposals)

            delta_e = []
            for p in proposals:
                if p.n_moved == 0:
                    delta_e.append(0.0)
                else:
                    de = energy_comp.compute_delta_energy(
                        p.chain_idx, p.bead_start, p.n_moved,
                        p.old_pos, p.new_pos, N)
                    delta_e.append(de)

            cl = energy_comp.cell_list
            nc = cl._nc; inv_cs = cl._inv_cs
            hb = cl.half_box; nc_m1 = nc - 1
            neighbor_cells_table = cl._neighbor_cells

            cell_nbr_sets = []
            for p in proposals:
                if p.n_moved == 0:
                    cell_nbr_sets.append(None)
                    continue
                cells = set()
                for k in range(p.n_moved):
                    cx = max(0, min(nc_m1, int((p.old_pos[k, 0] + hb) * inv_cs)))
                    cy = max(0, min(nc_m1, int((p.old_pos[k, 1] + hb) * inv_cs)))
                    cz = max(0, min(nc_m1, int((p.old_pos[k, 2] + hb) * inv_cs)))
                    cells.add((cx * nc + cy) * nc + cz)
                    cx = max(0, min(nc_m1, int((p.new_pos[k, 0] + hb) * inv_cs)))
                    cy = max(0, min(nc_m1, int((p.new_pos[k, 1] + hb) * inv_cs)))
                    cz = max(0, min(nc_m1, int((p.new_pos[k, 2] + hb) * inv_cs)))
                    cells.add((cx * nc + cy) * nc + cz)
                expanded = set()
                for c in cells:
                    expanded.update(neighbor_cells_table[c])
                cell_nbr_sets.append(expanded)

            r_rep_sq = cfg.r_rep_sq
            rep_e = cfg.repulsive_energy

            corr = [[0.0]*B for _ in range(B)]

            for i in range(B):
                if proposals[i].n_moved == 0:
                    continue
                nbr_i = cell_nbr_sets[i]
                for j in range(i + 1, B):
                    if proposals[j].n_moved == 0:
                        continue
                    nbr_j = cell_nbr_sets[j]
                    if nbr_i is not None and nbr_j is not None and nbr_i.isdisjoint(nbr_j):
                        continue
                    e00 = FastKernels.compute_segment_pair_energy(
                        proposals[i].old_pos, proposals[j].old_pos,
                        box, inv_box, r_rep_sq, rep_e)
                    e01 = FastKernels.compute_segment_pair_energy(
                        proposals[i].old_pos, proposals[j].new_pos,
                        box, inv_box, r_rep_sq, rep_e)
                    e10 = FastKernels.compute_segment_pair_energy(
                        proposals[i].new_pos, proposals[j].old_pos,
                        box, inv_box, r_rep_sq, rep_e)
                    e11 = FastKernels.compute_segment_pair_energy(
                        proposals[i].new_pos, proposals[j].new_pos,
                        box, inv_box, r_rep_sq, rep_e)
                    c = (e11 - e01) - (e10 - e00)
                    if c != 0.0:
                        corr[i][j] = c

            for i in range(B):
                if proposals[i].n_moved == 0:
                    continue

                accepted = FastProposer.metropolis_accept(delta_e[i], kBT, rng)
                stats.record(proposals[i].move_type, accepted)

                if not accepted:
                    continue

                p = proposals[i]
                positions_np[p.chain_idx, p.bead_start:p.bead_start + p.n_moved] = p.new_pos
                FastProposer.validate_boundary_bonds(
                    positions_np, p.chain_idx, p.bead_start,
                    p.n_moved, N, cfg.l0, box, inv_box)

                for j in range(i + 1, B):
                    if proposals[j].n_moved == 0:
                        continue
                    c = corr[i][j]
                    if c != 0.0:
                        delta_e[j] += c

        if skip_pivot:
            return

        pos_flat = positions_np.reshape(-1, 3)
        energy_comp.rebuild_cell_list(pos_flat)

        chain_perm = rng.permutation(n_chains)
        for c_idx in chain_perm:
            c = int(c_idx)
            chain_pos = positions_np[c]
            p = FastProposer.propose_pivot(chain_pos, c, N, box, inv_box, half_box, rng, cfg.l0)

            if p.n_moved == 0:
                continue

            de = energy_comp.compute_delta_energy(
                p.chain_idx, p.bead_start, p.n_moved,
                p.old_pos, p.new_pos, N)

            accepted = FastProposer.metropolis_accept(de, kBT, rng)
            stats.record('pivot', accepted)

            if accepted:
                positions_np[p.chain_idx, p.bead_start:p.bead_start + p.n_moved] = p.new_pos
                FastProposer.validate_boundary_bonds(
                    positions_np, p.chain_idx, p.bead_start,
                    p.n_moved, N, cfg.l0, box, inv_box)
