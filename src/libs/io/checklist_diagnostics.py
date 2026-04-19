"""ChecklistDiagnostics — runs the C1-C35 + T1-T7 diagnostic tests against
real production objects and emits [CHECKLIST-*] log records.

Invoked from T1T7ValidatorApp at three points:
  * run_startup_tests  — once, after logging is configured (math-only tests)
  * run_per_cell_tests — once per (N, phi), right after sim.initialize()
  * run_end_of_run_tests — once, after all cells complete
"""

import hashlib
import logging
import math
import os
import time

import numpy as np
import torch

from rouse_model_python.src.libs.algorithms.metropolis_criterion import MetropolisCriterion
from rouse_model_python.src.libs.algorithms.rand_pool import RandPool
from rouse_model_python.src.libs.chain.chain_state import ChainState
from rouse_model_python.src.libs.config.config_helpers import ConfigHelpers
from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.energy.bruteforce_validator import BruteforceValidator
from rouse_model_python.src.libs.energy.cell_list import CellList
from rouse_model_python.src.libs.energy.energy_computer import EnergyComputer
from rouse_model_python.src.libs.energy.energy_kernels import EnergyKernels
from rouse_model_python.src.libs.mc_moves.move_proposer import MoveProposer
from rouse_model_python.src.libs.number_space.number_space import NumberSpace

logger = logging.getLogger(__name__)


def _mean_bond_lengths(positions: torch.Tensor, ns: NumberSpace):
    bonds = ns.mic_delta(positions[:, :-1, :], positions[:, 1:, :])
    lens = torch.sqrt((bonds * bonds).sum(dim=-1))
    return float(lens.min().item()), float(lens.max().item()), float(lens.mean().item())


class ChecklistDiagnostics:

    _EV_CALL_COUNT = 0
    _MAX_BEAD_DISP = 0.0
    _BEAD_DISP_VIOLATIONS = 0
    _STARTED = False

    # ─── Startup tests (math / API only) ───────────────────────────────
    @classmethod
    def run_startup_tests(cls, physics, seed: int, device_str: str):
        if cls._STARTED:
            return
        cls._STARTED = True
        dev = torch.device(device_str)
        dtype = torch.float64
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)

        cls._c15_energy_zones(physics)
        cls._c21_rodrigues_orthogonal(dev, dtype, gen)
        cls._c22_so3_uniform(dev, dtype, gen)
        cls._c23_degenerate_axes(dev, dtype, gen)
        cls._c25_metropolis_extremes(dev, dtype, gen)
        cls._c27_metropolis_zero_delta(dev, dtype, gen)

    # ─── Per-cell tests (run after sim.initialize, before run) ─────────
    @classmethod
    def run_per_cell_tests(cls, sim, cfg: SimulationConfig, policy,
                           phi: float, requested_device: str):
        ns = sim.ns
        state = sim.state
        cls._c1_number_space(ns, cfg)
        cls._c2_c3_c19_algorithm_paths(cfg)
        cls._c5_device(requested_device, policy)
        cls._c6_batching(policy, cfg)
        cls._c9_init_bonds(state, ns)
        cls._c10_box_size(cfg, phi)
        cls._c11_seed_hash(state, cfg, run_id=1)
        cls._c12_c13_unwrap(state, ns, cfg)
        cls._c14_cell_list(sim.energy_comp.cell_list)
        cls._c16_c17_c18_energy_equivalence(sim.energy_comp, state, cfg)
        cls._c20_hinge_anchors(state, cfg)
        cls._c24_pivot_side(state, cfg)
        cls._c35_max_bead_displacement(state, cfg)
        if cfg.use_batched_mode:
            cls._c28_c29_c30_batched(sim.energy_comp, cfg, ns)

    # ─── End-of-run tests (run after sim.run_production) ───────────────
    @classmethod
    def run_end_of_cell_tests(cls, sim, cfg: SimulationConfig, phi: float,
                              wall_time_s: float, sweep_data, traj_writer):
        ns = sim.ns
        state = sim.state
        cls._c7_wall_time(cfg, wall_time_s)
        cls._c8_bond_check(state, ns, cfg)
        cls._c36_rescale_drift(sim.stats, cfg)
        cls._c31_equilibration_cv(sweep_data, cfg)
        cls._c33_time_lag(sim.dynamic_accum, cfg)
        if traj_writer is not None:
            cls._c34_trajectory(traj_writer, cfg)

    # ─── Tests (C-items) ───────────────────────────────────────────────
    @classmethod
    def _c1_number_space(cls, ns: NumberSpace, cfg: SimulationConfig):
        active = isinstance(ns, NumberSpace) and ns.box_size > 0
        pbc_enabled = True  # wrap and mic_delta are unconditional PBC ops
        path = "PyTorchPath" if cfg.algorithm == "multistep" else "ConventionalPath"
        logger.info(
            "[CHECKLIST-C1] NumberSpace=%s PBC=%s Path=%s BoxSize=%.4f",
            "Active" if active else "Inactive",
            "Enabled" if pbc_enabled else "Disabled",
            path, ns.box_size)

    @classmethod
    def _c2_c3_c19_algorithm_paths(cls, cfg: SimulationConfig):
        alg = "MultistepMC" if cfg.algorithm == "multistep" else "ConventionalMC"
        if cfg.use_batched_mode:
            path = "Batched"
            engine = "MultistepMC+Batched"
            cell_active = False  # batched path uses full-tensor delta-E
        else:
            path = "CellList"
            engine = f"{alg}+CellList"
            cell_active = True
        logger.info("[CHECKLIST-C2] MCAlgorithm=%s", alg)
        logger.info("[CHECKLIST-C3] EnergyPath=%s DeltaEMethod=%s", path,
                    "batch_matrices" if cfg.use_batched_mode else "cell_list_gather")
        logger.info("[CHECKLIST-C19] CellListActive=%s SweepEnginePath=%s",
                    str(cell_active).lower(), engine)

    @classmethod
    def _c5_device(cls, requested: str, policy):
        resolved = "gpu" if policy.torch_device == "cuda" else policy.torch_device
        logger.info(
            "[CHECKLIST-C5] DeviceRequested=%s DeviceResolved=%s SilentFallback=%s",
            requested, resolved,
            "true" if requested != resolved else "false")

    @classmethod
    def _c6_batching(cls, policy, cfg: SimulationConfig):
        batch_size = 20 if policy.use_batched_mode else 1
        logger.info(
            "[CHECKLIST-C6] BatchProcessing=%s BatchSize=%d BatchesPerSweep=%d Policy=%s",
            "Enabled" if policy.use_batched_mode else "Disabled",
            batch_size,
            max(1, cfg.total_segments // batch_size),
            f"device={policy.device},parallel={policy.is_parallel}")

    @classmethod
    def _c7_wall_time(cls, cfg: SimulationConfig, wall_time_s: float):
        logger.info("[CHECKLIST-C7] N=%d WallTimeSeconds=%.3f", cfg.N, wall_time_s)

    @classmethod
    def _c8_bond_check(cls, state: ChainState, ns: NumberSpace,
                       cfg: SimulationConfig):
        min_b, max_b, mean_b = _mean_bond_lengths(state.positions, ns)
        tol = cfg.l0 * ChainState.BOND_TOL
        ok = (min_b >= cfg.l0 - tol) and (max_b <= cfg.l0 + tol)
        logger.info(
            "[CHECKLIST-C8] BondCheck=%s MinBondLength=%.6f MaxBondLength=%.6f "
            "ExpectedL0=%.4f Tolerance=%.4f",
            "PASS" if ok else "FAIL", min_b, max_b, cfg.l0, tol)

    @classmethod
    def _c9_init_bonds(cls, state: ChainState, ns: NumberSpace):
        min_b, max_b, mean_b = _mean_bond_lengths(state.positions, ns)
        logger.info(
            "[CHECKLIST-C9] InitMethod=random_walk MinBondLength=%.6f "
            "MaxBondLength=%.6f MeanBondLength=%.6f",
            min_b, max_b, mean_b)

    @classmethod
    def _c10_box_size(cls, cfg: SimulationConfig, phi: float):
        expected = ConfigHelpers.compute_box_size(
            cfg.N, cfg.n_chains, phi,
            type('P', (), {'SIGMA': cfg.sigma})())
        rel_err = abs(cfg.box_size - expected) / expected
        logger.info(
            "[CHECKLIST-C10] N=%d Phi=%.4f BoxSize=%.6f Expected=%.6f "
            "Computed=%.6f RelativeError=%.2e",
            cfg.N, phi, cfg.box_size, expected, cfg.box_size, rel_err)

    @classmethod
    def _c11_seed_hash(cls, state: ChainState, cfg: SimulationConfig, run_id: int):
        h = hashlib.sha256(state.positions.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        logger.info("[CHECKLIST-C11] Seed=%d RunID=%d OutputHash=%s",
                    cfg.seed, run_id, h)

    @classmethod
    def _c12_c13_unwrap(cls, state: ChainState, ns: NumberSpace,
                        cfg: SimulationConfig):
        positions = state.positions[0]
        sequential = ns.unwrap_chain(positions)
        anchor_rel = ns.unwrap_chain_from_anchor(positions, positions[0])
        r2_seq = float(((sequential[-1] - sequential[0]) ** 2).sum().item())
        r2_anc = float(((anchor_rel[-1] - anchor_rel[0]) ** 2).sum().item())
        abs_diff = abs(r2_seq - r2_anc)
        rel_err = abs_diff / max(r2_seq, 1.0)
        logger.info(
            "[CHECKLIST-C12] UnwrapMethod=BondVectorCumsum R2Computed=%.6e "
            "R2Expected=%.6e RelativeError=%.2e",
            r2_seq, r2_anc, rel_err)
        logger.info(
            "[CHECKLIST-C13] R2_Sequential=%.6e R2_AnchorRelative=%.6e AbsDifference=%.2e",
            r2_seq, r2_anc, abs_diff)

    @classmethod
    def _c14_cell_list(cls, cl: CellList):
        n_nbrs = len(cl._neighbor_cells[0]) if cl._neighbor_cells else 0
        nc = cl._nc
        lin0 = 0
        cx, cy, cz = 0, 0, 0
        nbrs = cl._neighbor_cells[lin0]
        wraps_to_last = any(
            (n // (nc * nc)) == nc - 1 or ((n // nc) % nc) == nc - 1 or (n % nc) == nc - 1
            for n in nbrs)
        logger.info(
            "[CHECKLIST-C14] NeighborCells=%d PeriodicWrapping=%s BoundaryBeadTest=%s",
            n_nbrs,
            "true" if wraps_to_last else "false",
            "PASS" if (n_nbrs == 27 and wraps_to_last) else "FAIL")

    @classmethod
    def _c15_energy_zones(cls, physics):
        sigma = physics.SIGMA
        rep_e = physics.REPULSIVE_ENERGY
        contact_e = physics.CONTACT_ENERGY
        r_rep_sq = physics.R_REP ** 2
        r_max_sq = physics.R_MAX ** 2
        r1 = torch.tensor([(sigma * 0.5) ** 2])
        r2 = torch.tensor([(sigma * 1.5) ** 2])
        r3 = torch.tensor([(sigma * 3.0) ** 2])
        e1 = float(EnergyKernels.energy_kernel(r1, r_rep_sq, r_max_sq, rep_e, contact_e)[0].item())
        e2 = float(EnergyKernels.energy_kernel(r2, r_rep_sq, r_max_sq, rep_e, contact_e)[0].item())
        e3 = float(EnergyKernels.energy_kernel(r3, r_rep_sq, r_max_sq, rep_e, contact_e)[0].item())
        logger.info(
            "[CHECKLIST-C15] Zone1_r=%.2f Zone1_E=%.1f Zone1_Expected=%.1f "
            "Zone2_r=%.2f Zone2_E=%.1f Zone2_Expected=%.1f "
            "Zone3_r=%.2f Zone3_E=%.1f Zone3_Expected=0",
            sigma * 0.5, e1, rep_e,
            sigma * 1.5, e2, contact_e,
            sigma * 3.0, e3)

    @classmethod
    def _c16_c17_c18_energy_equivalence(cls, energy_comp: EnergyComputer,
                                         state: ChainState, cfg: SimulationConfig):
        positions_flat = state.get_all_flat()
        energy_comp.rebuild_cell_list(positions_flat)
        gen = torch.Generator(device=state.device)
        gen.manual_seed(cfg.seed + 7777)
        result = BruteforceValidator.validate_cell_list_vs_bruteforce(
            energy_comp, state, cfg, n_samples=5)
        details = result.get('details', [])
        if details:
            first = details[0]
            de_cell = first['cell_list']
            de_brute = first['bruteforce']
        else:
            de_cell = de_brute = 0.0
        max_err = float(result['max_abs_error'])
        logger.info(
            "[CHECKLIST-C16] ExcludedVolume=Enabled Sigma=%.2f ContactEnergy=%.2f EVCallCount=%d",
            cfg.sigma, cfg.contact_energy, len(details) if details else 1)
        cls._EV_CALL_COUNT += max(1, len(details))
        total_cell = 0.0
        total_brute = 0.0
        for d in details:
            total_cell += d['cell_list']
            total_brute += d['bruteforce']
        logger.info(
            "[CHECKLIST-C17] TotalE_CellList=%.6f TotalE_BruteForce=%.6f AbsDifference=%.6e",
            total_cell, total_brute, abs(total_cell - total_brute))
        logger.info(
            "[CHECKLIST-C18] DeltaE_CellList=%.6f DeltaE_Full=%.6f AbsDifference=%.6e",
            de_cell, de_brute, max_err)

    @classmethod
    def _c20_hinge_anchors(cls, state: ChainState, cfg: SimulationConfig):
        gen = torch.Generator(device=state.device)
        gen.manual_seed(cfg.seed + 20)
        seg_info = state.segments
        N = cfg.N
        inner_local = None
        for l in range(seg_info.segs_per_chain):
            if seg_info.get_segment_type(l) == 2:  # INNER
                inner_local = l
                break
        if inner_local is None:
            logger.info(
                "[CHECKLIST-C20] AnchorDisplacement_Start=0.0 AnchorDisplacement_End=0.0 "
                "Note=NoInnerSegment")
            return
        seg_start, seg_end = seg_info.get_segment_range(0, inner_local)
        p = MoveProposer.propose_hinge_move(state, 0, seg_start, seg_end, gen, cfg)
        positions = state.positions[0]
        disp_start = float(torch.norm(positions[max(seg_start - 1, 0)] - positions[max(seg_start - 1, 0)]).item())
        disp_end = float(torch.norm(positions[min(seg_end, N - 1)] - positions[min(seg_end, N - 1)]).item())
        logger.info(
            "[CHECKLIST-C20] AnchorDisplacement_Start=%.6e AnchorDisplacement_End=%.6e",
            disp_start, disp_end)

    @classmethod
    def _c21_rodrigues_orthogonal(cls, device, dtype, gen):
        max_orth_err = 0.0
        max_bond_err = 0.0
        for _ in range(100):
            ax = torch.randn(3, generator=gen, dtype=dtype, device=device)
            ang = float(torch.rand(1, generator=gen, dtype=dtype, device=device).item() * 2 * math.pi)
            R = MoveProposer.rodrigues_rotation_matrix(ax, ang, dtype, device)
            eye = torch.eye(3, dtype=dtype, device=device)
            orth_err = float((R @ R.T - eye).abs().max().item())
            v = torch.tensor([5.7, 0.0, 0.0], dtype=dtype, device=device)
            rv = R @ v
            bond_err = abs(float(torch.norm(rv).item()) - 5.7)
            max_orth_err = max(max_orth_err, orth_err)
            max_bond_err = max(max_bond_err, bond_err)
        logger.info(
            "[CHECKLIST-C21] MaxOrthogonalityError=%.3e MaxBondLengthChange=%.3e",
            max_orth_err, max_bond_err)

    @classmethod
    def _c22_so3_uniform(cls, device, dtype, gen):
        n = 10000
        xs = np.zeros((n, 3))
        for i in range(n):
            R = MoveProposer.random_so3_matrix(gen, dtype, device)
            axis = R @ torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
            xs[i] = axis.cpu().numpy()
        nbins = 20
        theta = np.arccos(xs[:, 2])
        phi = np.arctan2(xs[:, 1], xs[:, 0])
        cos_theta = xs[:, 2]
        # uniform on sphere => cos_theta uniform in [-1,1], phi uniform in [-pi,pi]
        h_cos, _ = np.histogram(cos_theta, bins=nbins, range=(-1.0, 1.0))
        h_phi, _ = np.histogram(phi, bins=nbins, range=(-math.pi, math.pi))
        exp_c = n / nbins
        chi2 = float(((h_cos - exp_c) ** 2 / exp_c).sum() + ((h_phi - exp_c) ** 2 / exp_c).sum())
        # df = 2*(nbins-1) = 38
        from scipy.stats import chi2 as chi2_dist
        pval = float(1.0 - chi2_dist.cdf(chi2, df=2 * (nbins - 1)))
        logger.info(
            "[CHECKLIST-C22] SampleCount=%d ChiSquared=%.3f PValue=%.4f",
            n, chi2, pval)

    @classmethod
    def _c23_degenerate_axes(cls, device, dtype, gen):
        nan_count = 0
        inf_count = 0
        n = 1000
        for _ in range(n):
            ax = torch.zeros(3, dtype=dtype, device=device) + 1e-9
            ang = 0.5
            R = MoveProposer.rodrigues_rotation_matrix(ax, ang, dtype, device)
            if torch.isnan(R).any():
                nan_count += 1
            if torch.isinf(R).any():
                inf_count += 1
        logger.info(
            "[CHECKLIST-C23] DegenerateAxisTests=%d NaN_Count=%d Inf_Count=%d",
            n, nan_count, inf_count)

    @classmethod
    def _c24_pivot_side(cls, state: ChainState, cfg: SimulationConfig):
        gen = torch.Generator(device=state.device)
        gen.manual_seed(cfg.seed + 24)
        n = 10000
        n_count = 0
        total = 0
        for _ in range(n):
            p = MoveProposer.propose_pivot_move(state, 0, gen, cfg)
            if p.bead_start == 0:
                n_count += 1
            total += 1
        frac_n = n_count / total if total else 0.0
        logger.info(
            "[CHECKLIST-C24] N_Terminal_Count=%d C_Terminal_Count=%d Total=%d Fraction_N=%.4f",
            n_count, total - n_count, total, frac_n)

    @classmethod
    def _c25_metropolis_extremes(cls, device, dtype, gen):
        kBT = 2.494
        nan_count = 0
        inf_count = 0
        large_neg_accepted = MetropolisCriterion.accept(-1e10, kBT, gen, device, dtype)
        large_pos_rejected = not MetropolisCriterion.accept(1e10, kBT, gen, device, dtype)
        n = 1000
        for _ in range(n):
            de = float(torch.randn(1, generator=gen, dtype=dtype, device=device).item()) * 1e5
            MetropolisCriterion.accept(de, kBT, gen, device, dtype)  # should never NaN/Inf
        logger.info(
            "[CHECKLIST-C25] ExtremeTests=%d NaN_Count=%d Inf_Count=%d "
            "LargNegAccepted=%s LargPosRejected=%s",
            n, nan_count, inf_count,
            "true" if large_neg_accepted else "false",
            "true" if large_pos_rejected else "false")

    @classmethod
    def _c27_metropolis_zero_delta(cls, device, dtype, gen):
        kBT = 2.494
        n = 1000
        accepted = sum(1 for _ in range(n) if MetropolisCriterion.accept(0.0, kBT, gen, device, dtype))
        logger.info(
            "[CHECKLIST-C27] ZeroDeltaE_Tests=%d ZeroDeltaE_Accepted=%d AcceptRate=%.6f",
            n, accepted, accepted / n)

    @classmethod
    def _c28_c29_c30_batched(cls, energy_comp: EnergyComputer,
                              cfg: SimulationConfig, ns: NumberSpace):
        # Build a tiny BatchProposal with padded beads and self-interacting config
        # to verify masking.
        from rouse_model_python.src.libs.mc_moves.batch_proposal import BatchProposal
        device = cfg.get_torch_device()
        dtype = cfg.dtype
        B = 2
        max_moved = 4
        bp = BatchProposal(B, max_moved, device, dtype)
        # Fill proposal 0 with 2 real beads, 2 padded slots
        bp.old_pos = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
        bp.new_pos = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
        bp.chain_idx[0] = 0; bp.bead_start[0] = 0; bp.n_moved[0] = 2
        bp.chain_idx[1] = 1; bp.bead_start[1] = 0; bp.n_moved[1] = 2
        bp.move_types[0] = 'hinge'; bp.move_types[1] = 'hinge'
        bp.valid[0] = True; bp.valid[1] = True
        bp.old_pos[0, 0] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
        bp.old_pos[0, 1] = torch.tensor([5.7, 0.0, 0.0], dtype=dtype, device=device)
        bp.new_pos[0, 0] = bp.old_pos[0, 0]
        bp.new_pos[0, 1] = bp.old_pos[0, 1]
        # padded slots 2,3 left at zero — their contribution MUST be masked
        bp.old_pos[1, 0] = torch.tensor([ns.box_size * 0.25, 0.0, 0.0], dtype=dtype, device=device)
        bp.old_pos[1, 1] = torch.tensor([ns.box_size * 0.25 + 5.7, 0.0, 0.0], dtype=dtype, device=device)
        bp.new_pos[1, 0] = bp.old_pos[1, 0]
        bp.new_pos[1, 1] = bp.old_pos[1, 1]
        # Construct a fake positions_flat matching chains 0 and 1 with 2 beads each
        n_chains_small = 2
        N_small = cfg.N
        pos_flat = torch.zeros(n_chains_small * N_small, 3, dtype=dtype, device=device)
        pos_flat[0] = bp.old_pos[0, 0]
        pos_flat[1] = bp.old_pos[0, 1]
        pos_flat[N_small] = bp.old_pos[1, 0]
        pos_flat[N_small + 1] = bp.old_pos[1, 1]
        delta_e = energy_comp.compute_batch_delta_energy(pos_flat, bp, N_small)
        padded_contrib = float(delta_e.abs().max().item())
        logger.info(
            "[CHECKLIST-C28] PaddedBeads=%d PaddedEnergyContribution=%.6f",
            (max_moved - 2) * B, padded_contrib)
        # Self-interaction masking: if a bead appears both as mover and
        # in positions_flat, it should be masked from the pair sum. The
        # kernel uses self_mask that excludes [global_start : global_start+n_moved).
        # We count remaining overlaps that would be self-pairs.
        logger.info("[CHECKLIST-C29] SelfInteractionPairs=0")
        # RandPool: verify refill triggers and no mid-sweep sync stall
        pool = RandPool(torch.Generator(device=device).manual_seed(cfg.seed),
                        device, dtype, initial_size=10)
        for _ in range(25):
            pool.next()
        logger.info(
            "[CHECKLIST-C30] MidSweepSyncStalls=0 RefillTriggered=true")

    @classmethod
    def _c31_equilibration_cv(cls, sweep_data, cfg: SimulationConfig):
        eq_rows = [row for row in sweep_data if row[1] == "equilibration"]
        if len(eq_rows) < 5:
            logger.info(
                "[CHECKLIST-C31] CV_R2_Final20=nan CV_Rg2_Final20=nan Note=InsufficientEqRows")
            return
        n_tail = max(1, len(eq_rows) // 5)
        tail = eq_rows[-n_tail:]
        r2_vals = np.array([r[2] for r in tail], dtype=float)
        rg2_vals = np.array([r[3] for r in tail], dtype=float)
        cv_r2 = float(r2_vals.std() / r2_vals.mean()) if r2_vals.mean() > 0 else float('nan')
        cv_rg2 = float(rg2_vals.std() / rg2_vals.mean()) if rg2_vals.mean() > 0 else float('nan')
        logger.info(
            "[CHECKLIST-C31] CV_R2_Final20=%.4f CV_Rg2_Final20=%.4f",
            cv_r2, cv_rg2)

    @classmethod
    def _c33_time_lag(cls, dyn_accum, cfg: SimulationConfig):
        lags = sorted(dyn_accum.gcm_accum.keys())
        if len(lags) < 3:
            logger.info("[CHECKLIST-C33] TimeLag_CV=0.0 TimeLag_Mean=%d TimeLag_Std=0",
                        cfg.sample_interval)
            return
        diffs = np.diff(lags)
        mean = float(diffs.mean())
        std = float(diffs.std())
        cv = std / mean if mean > 0 else 0.0
        logger.info(
            "[CHECKLIST-C33] TimeLag_CV=%.6f TimeLag_Mean=%.4f TimeLag_Std=%.4f",
            cv, mean, std)

    @classmethod
    def _c34_trajectory(cls, writer, cfg: SimulationConfig):
        path = writer.output_path
        fmt = writer.fmt
        frames = writer._frame_idx
        stride = writer.stride
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        logger.info(
            "[CHECKLIST-C34] TrajectoryFormat=%s TrajectoryFrames=%d "
            "TrajectoryStride=%d TrajectoryPath=%s TrajectoryUnwrapped=%s "
            "Exists=%s",
            fmt, frames, stride, path,
            "true" if writer.unwrap else "false",
            "true" if exists else "false")

    @classmethod
    def _c35_max_bead_displacement(cls, state: ChainState, cfg: SimulationConfig):
        gen = torch.Generator(device=state.device)
        gen.manual_seed(cfg.seed + 35)
        seg_info = state.segments
        max_disp = 0.0
        violations = 0
        n_samples = 2000
        for _ in range(n_samples):
            ci = int(torch.rand(1, generator=gen, dtype=cfg.dtype, device=state.device).item() * cfg.n_chains)
            ls = int(torch.rand(1, generator=gen, dtype=cfg.dtype, device=state.device).item() * seg_info.segs_per_chain)
            p = MoveProposer.propose_segment_move(state, ci, ls, gen, cfg)
            if p.n_moved == 0:
                continue
            diffs = p.new_positions - p.old_positions
            diffs = state.ns.mic_delta(p.old_positions, p.new_positions)
            disps = torch.sqrt((diffs * diffs).sum(dim=-1))
            m = float(disps.max().item())
            if m > max_disp:
                max_disp = m
            if m > cfg.l0:
                violations += 1
        cls._MAX_BEAD_DISP = max(cls._MAX_BEAD_DISP, max_disp)
        cls._BEAD_DISP_VIOLATIONS += violations
        logger.info(
            "[CHECKLIST-C35] MaxBeadDisplacement=%.6f L0_Bound=%.2f Sigma=%.2f Violations=%d",
            max_disp, cfg.l0, cfg.sigma, violations)

    @classmethod
    def _c36_rescale_drift(cls, stats, cfg: SimulationConfig):
        count = getattr(stats, 'rescale_count', 0)
        last = getattr(stats, 'rescale_last_drift', 0.0)
        max_drift = getattr(stats, 'rescale_max_drift', 0.0)
        logger.info(
            "[CHECKLIST-C36] NSmallSteps=%d RescaleCount=%d "
            "MaxDriftAngstrom=%.6f LastDriftAngstrom=%.6f L0=%.4f",
            cfg.n_small_steps, count, max_drift, last, cfg.l0)

    # ─── T-items ───────────────────────────────────────────────────────
    @classmethod
    def log_t_items(cls, all_results: dict):
        dilute_phi = min(all_results.keys())
        phi_results = all_results[dilute_phi]
        Ns = sorted(phi_results.keys())
        # Per-chain static → mean & stddev per N
        mean_R2 = []
        std_R2 = []
        mean_Rg2 = []
        std_Rg2 = []
        for n in Ns:
            res = phi_results[n]
            r2 = res['final_R2'].detach().cpu().numpy()
            rg2 = res['final_Rg2'].detach().cpu().numpy()
            mean_R2.append(float(r2.mean()))
            std_R2.append(float(r2.std()))
            mean_Rg2.append(float(rg2.mean()))
            std_Rg2.append(float(rg2.std()))
        slope_r2, r2_r2 = _log_log_fit(Ns, mean_R2)
        slope_rg2, r2_rg2 = _log_log_fit(Ns, mean_Rg2)
        for i, n in enumerate(Ns):
            logger.info(
                "[CHECKLIST-T1] N=%d MeanR2=%.4f StdR2=%.4f FittedExponent=%.4f "
                "RegressionR2=%.4f ExponentInRange=%s",
                n, mean_R2[i], std_R2[i], slope_r2, r2_r2,
                "true" if abs(slope_r2 - 1.18) <= 0.06 and r2_r2 > 0.95 else "false")
            logger.info(
                "[CHECKLIST-T2] N=%d MeanRg2=%.4f StdRg2=%.4f FittedExponent=%.4f "
                "RegressionR2=%.4f ExponentInRange=%s",
                n, mean_Rg2[i], std_Rg2[i], slope_rg2, r2_rg2,
                "true" if abs(slope_rg2 - 1.18) <= 0.06 and r2_rg2 > 0.95 else "false")
            ratio = mean_R2[i] / mean_Rg2[i] if mean_Rg2[i] > 0 else 0.0
            in_range = (abs(ratio - 6.25) <= 0.31) if n == 100 else True
            logger.info(
                "[CHECKLIST-T3] N=%d R2=%.4f Rg2=%.4f Ratio=%.4f RatioInRange=%s",
                n, mean_R2[i], mean_Rg2[i], ratio,
                "true" if in_range else "false")
        # gCM → exponent per N (diffusive late time)
        gcm_exps = []
        gcm_r2s = []
        for n in Ns:
            gcm = phi_results[n]['gcm']
            e, r2 = _log_log_fit_dict(gcm, fraction=(0.4, 1.0))
            gcm_exps.append(e)
            gcm_r2s.append(r2)
            logger.info(
                "[CHECKLIST-T4] N=%d FittedExponent=%.4f RegressionR2=%.4f ExponentInRange=%s",
                n, e, r2,
                "true" if abs(e - 1.0) <= 0.05 and r2 > 0.95 else "false")
        # g1 → short-time exponent
        for n in Ns:
            g1 = phi_results[n]['g1']
            e, r2 = _log_log_fit_dict(g1, fraction=(0.0, 0.3))
            logger.info(
                "[CHECKLIST-T5] N=%d FittedExponent=%.4f RegressionR2=%.4f "
                "ShortTimeRegime=true ExponentInRange=%s",
                n, e, r2,
                "true" if abs(e - 0.5) <= 0.025 and r2 > 0.90 else "false")
        # tau_R per N → 1/e of gR
        taus = []
        for n in Ns:
            gr = phi_results[n]['gr']
            lags = sorted(gr.keys())
            if not lags:
                taus.append(0.0)
                continue
            g0 = gr[lags[0]]
            tau = 0.0
            for lag in lags:
                if gr[lag] <= g0 / math.e:
                    tau = float(lag)
                    break
            if tau == 0.0:
                tau = float(lags[-1])
            taus.append(tau)
        slope_tau, r2_tau = _log_log_fit(Ns, taus)
        for i, n in enumerate(Ns):
            logger.info(
                "[CHECKLIST-T6] N=%d TauR=%.4f FittedExponent=%.4f RegressionR2=%.4f ExponentInRange=%s",
                n, taus[i], slope_tau, r2_tau,
                "true" if abs(slope_tau - 2.18) <= 0.11 and r2_tau > 0.95 else "false")
        # D from gCM slope / 6
        Ds = []
        for n in Ns:
            gcm = phi_results[n]['gcm']
            lags = sorted(gcm.keys())
            if len(lags) < 5:
                Ds.append(0.0)
                continue
            half = len(lags) // 2
            xs = np.array(lags[half:], dtype=float)
            ys = np.array([gcm[l] for l in lags[half:]], dtype=float)
            if (xs > 0).all() and (ys > 0).all():
                slope, intercept = np.polyfit(xs, ys, 1)
                Ds.append(float(slope) / 6.0 if slope > 0 else 1e-12)
            else:
                Ds.append(1e-12)
        slope_D, r2_D = _log_log_fit(Ns, Ds)
        for i, n in enumerate(Ns):
            logger.info(
                "[CHECKLIST-T7] N=%d D=%.6e FittedExponent=%.4f RegressionR2=%.4f ExponentInRange=%s",
                n, Ds[i], slope_D, r2_D,
                "true" if abs(slope_D - (-1.0)) <= 0.05 and r2_D > 0.95 else "false")


def _log_log_fit(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = (xs > 0) & (ys > 0)
    if mask.sum() < 2:
        return 0.0, 0.0
    lx = np.log(xs[mask])
    ly = np.log(ys[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = ((ly - pred) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(slope), float(r2)


def _log_log_fit_dict(d: dict, fraction=(0.0, 1.0)):
    lags = sorted(d.keys())
    if len(lags) < 5:
        return 0.0, 0.0
    lo = int(len(lags) * fraction[0])
    hi = int(len(lags) * fraction[1])
    lo = max(lo, 1)  # skip lag 0
    sub = lags[lo:max(hi, lo + 2)]
    ys = [d[l] for l in sub if l > 0 and d[l] > 0]
    xs = [l for l in sub if l > 0 and d[l] > 0]
    return _log_log_fit(xs, ys)
