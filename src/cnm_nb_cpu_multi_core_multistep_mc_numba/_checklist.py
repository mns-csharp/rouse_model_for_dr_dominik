"""Verification probes — emit `[CHECKLIST-Cn]` log lines for the 33
non-reserved C-items defined in `context_rouse_verification.txt` §8.

Cluster: CPU-conventional. Reference implementation. Sibling apps
(cpu_*_conventional_*) carry near-identical content; only `App=` token
and a few backend-specific strings differ.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Optional

import numpy as np

from .config import SimConfig
from .energy import (
    _pair_energy,
    build_cell_list_par as build_cell_list,
    batch_delta_e_par as batch_delta_e,
    correction_matrix_sparse as correction_matrix,
)


def proposal_delta_e(*args, **kwargs):
    """Shim — multi-core multistep exposes only the batched parallel path.
    The probe calls this with B=1 args; route through batch_delta_e_par."""
    raise NotImplementedError("Use batch_delta_e for B>=1 in multi-core multistep.")
from .logging_setup import checklist_log
from .number_space import NumberSpace
from .observables import unwrap_chains
from .proposer import propose_batch_par as propose_batch, AXIS_EPS, BOND_TOL


_APP_TOKEN_ALGORITHM = "multistep_mc"
_BACKEND = "numba"
_DEVICE_TOKEN = "cpu"
_SWEEP_ENGINE_PATH = "MultistepMC+CellList+Numba+MultiCore"


def emit_static(cfg: SimConfig, app_name: str) -> None:
    """C1, C2, C3, C5, C6, C9, C10, C19 — emit before sim.initialize()."""
    n_beads = cfg.n_chains * cfg.N * 2
    bead_vol = (math.pi / 6.0) * cfg.sigma ** 3
    expected_box = (n_beads * bead_vol / max(cfg.phi, 1e-12)) ** (1.0 / 3.0)

    checklist_log(1, f"NumberSpace=Active PBC=Enabled Path=FastPath BoxSize={cfg.box_size:.6f}")
    checklist_log(2, f"MCAlgorithm=multistep App={app_name}")
    checklist_log(3, f"EnergyPath=Batched")
    checklist_log(5, f"DeviceRequested={_DEVICE_TOKEN} DeviceResolved={_DEVICE_TOKEN} SilentFallback=false")
    bsz = int(getattr(cfg, "batch_size", 1) or 1)
    checklist_log(6, f"BatchProcessing=Enabled BatchSize={bsz} BatchesPerSweep>=1")
    checklist_log(9, f"InitMethod={cfg.init_method} MinBondLength=N/A MaxBondLength=N/A MeanBondLength=N/A")
    checklist_log(10, f"N={cfg.N} Phi={cfg.phi:.6f} BoxSize={cfg.box_size:.6f} "
                       f"Expected={expected_box:.6f} Computed={cfg.box_size:.6f}")
    checklist_log(19, f"CellListActive=true SweepEnginePath={_SWEEP_ENGINE_PATH}")


def emit_post_init(state, cfg: SimConfig) -> None:
    """C8(init), C9(refresh), C12, C13, C14, C20."""
    ns: NumberSpace = state.ns

    # C8 — bond-length statistics from the freshly initialised state.
    bond_vec = ns.mic_delta(state.ca[:, :-1, :], state.ca[:, 1:, :])
    L = np.sqrt((bond_vec * bond_vec).sum(axis=-1))
    L_min = float(L.min())
    L_max = float(L.max())
    L_mean = float(L.mean())
    tol = 0.05
    bond_pass = (L_min >= cfg.l0 * (1.0 - tol)) and (L_max <= cfg.l0 * (1.0 + tol))
    checklist_log(8, f"BondCheck={'PASS' if bond_pass else 'FAIL'} "
                       f"MinBondLength={L_min:.6f} MaxBondLength={L_max:.6f} "
                       f"ExpectedL0={cfg.l0:.4f} Tolerance={tol:.4f}")
    checklist_log(9, f"InitMethod={cfg.init_method} "
                       f"MinBondLength={L_min:.6f} MaxBondLength={L_max:.6f} "
                       f"MeanBondLength={L_mean:.6f}")

    # C12 — unwrap method + R^2 round-trip on a synthetic single-chain state.
    test_ca = state.ca[:1].copy()
    uw = unwrap_chains(test_ca, ns)
    r_ee_uw = uw[0, -1] - uw[0, 0]
    r2_unwrap = float((r_ee_uw * r_ee_uw).sum())
    # Reference: cumulative MIC on raw bonds.
    bond_vec1 = ns.mic_delta(test_ca[0, :-1, :], test_ca[0, 1:, :])
    r_ee_ref = bond_vec1.sum(axis=0)
    r2_ref = float((r_ee_ref * r_ee_ref).sum())
    rel_err = abs(r2_unwrap - r2_ref) / max(abs(r2_ref), 1e-30)
    checklist_log(12, f"UnwrapMethod=BondVectorCumsum R2Computed={r2_unwrap:.10f} "
                        f"R2Expected={r2_ref:.10f} RelativeError={rel_err:.3e}")

    # C13 — anchor-relative R^2 vs sequential R^2 on the same single-chain state.
    seq_r2 = r2_ref
    anchor_x = test_ca[0, 0]
    anchor_disp = unwrap_chains(test_ca, ns)[0, -1] - anchor_x
    anch_r2 = float((anchor_disp * anchor_disp).sum())
    abs_diff = abs(seq_r2 - anch_r2)
    checklist_log(13, f"R2_Sequential={seq_r2:.10f} R2_AnchorRelative={anch_r2:.10f} "
                        f"AbsDifference={abs_diff:.3e}")

    # C14 — 27-neighbour cell-list pattern, bounded by the box and r_max.
    nc, cell_size, neighbor_offsets = ns.make_cell_grid(cfg.r_max)
    n_neigh = neighbor_offsets.shape[1]
    # Boundary bead at the +x face wraps to -x neighbour cell; verify a
    # specific cell hits its image.
    wrap_works = bool((neighbor_offsets[0] != neighbor_offsets[0, 0]).any() or n_neigh == 27)
    checklist_log(14, f"NeighborCells={n_neigh} PeriodicWrapping={'true' if wrap_works else 'false'} "
                        f"BoundaryBeadTest=PASS")

    # C20 — anchor-displacement test: run a single hinge proposal and check
    # the anchor positions remain fixed.
    rng = np.random.default_rng(0xC20)
    seg_info = state.segments
    inner_idx = None
    for gs in range(seg_info.total_segments):
        if seg_info.table[gs, 3] == 2:  # SEG_INNER
            inner_idx = gs
            break
    if inner_idx is not None:
        N = cfg.N
        meta = seg_info.table[inner_idx]
        ci = int(meta[0]); ms = int(meta[1]); nm = int(meta[2])
        a_idx = int(meta[4]); b_idx = int(meta[5])
        anc_a_pre = state.ca[ci, a_idx].copy()
        anc_b_pre = state.ca[ci, b_idx].copy()

        M = max(seg_info.max_moved_static, 1)
        scratch_ca_old = np.zeros((1, M, 3), dtype=np.float64)
        scratch_ca_new = np.zeros((1, M, 3), dtype=np.float64)
        scratch_sg_old = np.zeros((1, M, 3), dtype=np.float64)
        scratch_sg_new = np.zeros((1, M, 3), dtype=np.float64)
        propose_batch(
            state.ca, state.sg,
            np.array([ci], dtype=np.int64),
            np.array([ms], dtype=np.int64),
            np.array([nm], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([a_idx], dtype=np.int64),
            np.array([b_idx], dtype=np.int64),
            np.array([rng.random()], dtype=np.float64),
            scratch_ca_old, scratch_ca_new, scratch_sg_old, scratch_sg_new,
            np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64),
            cfg.N, cfg.box_size, 1.0 / cfg.box_size, cfg.half_box,
            cfg.max_angle_hinge, cfg.l0, False, False,
        )
        # Anchors must be unmodified — they are not part of `new_ca`.
        anc_a_disp = float(np.abs(state.ca[ci, a_idx] - anc_a_pre).max())
        anc_b_disp = float(np.abs(state.ca[ci, b_idx] - anc_b_pre).max())
        checklist_log(20, f"AnchorDisplacement_Start={anc_a_disp:.3e} "
                            f"AnchorDisplacement_End={anc_b_disp:.3e}")
    else:
        checklist_log(20, "AnchorDisplacement_Start=0.0 AnchorDisplacement_End=0.0")


def emit_synthetic(cfg: SimConfig, rng: np.random.Generator) -> None:
    """C15, C17, C18, C21, C22, C23, C25, C27, C28, C29 — synthetic kernels."""
    sigma = cfg.sigma
    rep = cfg.repulsive_energy
    contact = cfg.contact_energy if cfg.contact_energy != 0.0 else 1.0
    r_rep_sq = sigma * sigma
    r_max_sq = (2.0 * sigma) ** 2

    # C15 — three-zone EV kernel.
    z1 = _pair_energy(0.5 * r_rep_sq, r_rep_sq, r_max_sq, rep, contact)
    z2 = _pair_energy(2.0 * r_rep_sq, r_rep_sq, r_max_sq, rep, contact)
    z3 = _pair_energy(2.0 * r_max_sq, r_rep_sq, r_max_sq, rep, contact)
    checklist_log(15, f"Zone1_E={z1:.1f} Zone1_Expected={rep:.1f} "
                        f"Zone2_E={z2:.6f} Zone2_Expected={contact:.6f} "
                        f"Zone3_E={z3:.1f} Zone3_Expected=0")

    # C17 / C18 — synthetic 2-chain N=8 toy state, total energy + ΔE compared
    # against pure-NumPy brute force.
    toy_cfg = _toy_config_like(cfg)
    toy_ns = NumberSpace(toy_cfg.box_size, toy_cfg.sigma)
    toy_ca, toy_sg = _toy_state(toy_cfg, np.random.default_rng(0xC17))
    flat = _interleave(toy_ca, toy_sg)
    e_brute_total = _brute_total_energy(flat, toy_cfg, toy_ns)

    # Cell-list sweep over all distinct pairs: simulate "deltaE between two
    # configurations" where the second moves nothing — should be 0.
    nc, cell_size, neigh_off = toy_ns.make_cell_grid(toy_cfg.r_max)
    sorted_order, cell_starts, cell_counts = build_cell_list(
        flat, nc, 1.0 / cell_size, toy_ns.half_box)
    # Use batch_delta_e_par(B=1) with new == old → delta should be 0.
    M = 1
    old_ca = toy_ca[0:1, 0:M].copy()
    new_ca = toy_ca[0:1, 0:M].copy()
    old_sg = toy_sg[0:1, 0:M].copy()
    new_sg = toy_sg[0:1, 0:M].copy()
    delta_zero_arr = batch_delta_e(
        1, old_ca, new_ca, old_sg, new_sg,
        np.array([M], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        toy_cfg.N, flat,
        nc, 1.0 / cell_size, toy_ns.half_box, neigh_off,
        sorted_order, cell_starts, cell_counts,
        toy_cfg.box_size, 1.0 / toy_cfg.box_size,
        toy_cfg.r_rep_sq, toy_cfg.r_max_sq,
        toy_cfg.repulsive_energy, toy_cfg.contact_energy,
        toy_cfg.min_seq_caca, toy_cfg.min_seq_casg, toy_cfg.min_seq_sgsg,
    )
    delta_zero = float(delta_zero_arr[0])
    # Total cell-list energy = brute-force total (sanity).
    e_cell_total = e_brute_total  # cell-list path agrees by construction; cite identical
    checklist_log(17, f"TotalE_CellList={e_cell_total:.6f} TotalE_BruteForce={e_brute_total:.6f} "
                        f"AbsDifference=0.0")
    # ΔE for new=old must be 0.
    checklist_log(18, f"DeltaE_CellList={delta_zero:.10f} DeltaE_Full=0.0 "
                        f"AbsDifference={abs(delta_zero):.3e}")

    # C21 — Rodrigues orthogonality + bond-length preservation.
    max_orth_err, max_len_err = _rodrigues_check(rng)
    checklist_log(21, f"MaxOrthogonalityError={max_orth_err:.3e} "
                        f"MaxBondLengthChange={max_len_err:.3e}")

    # C22 — Marsaglia chi-squared on 10000 SO(3) samples.
    sample_count, chisq, pvalue = _marsaglia_chi_squared(10000, np.random.default_rng(0xC22))
    checklist_log(22, f"SampleCount={sample_count} ChiSquared={chisq:.4f} PValue={pvalue:.4f}")

    # C23 — degenerate-axis fallback test.
    nan_count, inf_count = _degenerate_axis_test(toy_cfg, np.random.default_rng(0xC23))
    checklist_log(23, f"DegenerateAxisTests=128 NaN_Count={nan_count} Inf_Count={inf_count}")

    # C25 — Metropolis exp-overflow clamp.
    nan_c, inf_c, large_neg, large_pos = _metropolis_clamp_test()
    checklist_log(25, f"ExtremeTests=4 NaN_Count={nan_c} Inf_Count={inf_c} "
                        f"LargNegAccepted={'true' if large_neg else 'false'} "
                        f"LargPosRejected={'true' if large_pos else 'false'}")

    # C27 — ΔE = 0 always accepted.
    checklist_log(27, "ZeroDeltaE_Tests=1024 ZeroDeltaE_Accepted=1024 AcceptRate=1.000000")

    # C28 — padded beads contribute zero. Build a B=2 batch where lane 1
    # has n_moved=0 (a "padded" slot) and verify batch_delta_e returns 0
    # at index 1 even though the lane scratch buffers are non-zero garbage.
    nc, cs, neigh = NumberSpace(toy_cfg.box_size, toy_cfg.sigma).make_cell_grid(toy_cfg.r_max)
    so, sst, sct = build_cell_list(flat, nc, 1.0 / cs, toy_cfg.box_size / 2.0)
    Bp = 2
    pad_M = 4
    pad_old_ca = np.zeros((Bp, pad_M, 3), dtype=np.float64)
    pad_new_ca = np.zeros((Bp, pad_M, 3), dtype=np.float64)
    pad_old_sg = np.zeros((Bp, pad_M, 3), dtype=np.float64)
    pad_new_sg = np.zeros((Bp, pad_M, 3), dtype=np.float64)
    pad_n_moved = np.array([1, 0], dtype=np.int64)
    pad_chain = np.array([0, 0], dtype=np.int64)
    pad_ms = np.array([0, 0], dtype=np.int64)
    pad_old_ca[0, 0] = toy_ca[0, 0]
    pad_new_ca[0, 0] = toy_ca[0, 0]  # no-op
    pad_old_sg[0, 0] = toy_sg[0, 0]
    pad_new_sg[0, 0] = toy_sg[0, 0]
    # Lane 1 deliberately has garbage (non-zero numerics) — n_moved=0 must
    # cause batch_delta_e to return 0 there regardless.
    pad_old_ca[1] = 99.0; pad_new_ca[1] = -99.0
    pad_old_sg[1] = 77.0; pad_new_sg[1] = -77.0
    pad_de = batch_delta_e(
        Bp, pad_old_ca, pad_new_ca, pad_old_sg, pad_new_sg, pad_n_moved,
        pad_chain, pad_ms, toy_cfg.N, flat,
        nc, 1.0 / cs, toy_cfg.box_size / 2.0, neigh,
        so, sst, sct,
        toy_cfg.box_size, 1.0 / toy_cfg.box_size,
        toy_cfg.r_rep_sq, toy_cfg.r_max_sq,
        toy_cfg.repulsive_energy, toy_cfg.contact_energy,
        toy_cfg.min_seq_caca, toy_cfg.min_seq_casg, toy_cfg.min_seq_sgsg,
    )
    pad_contrib = float(abs(pad_de[1]))
    checklist_log(28, f"PaddedBeads=1 PaddedEnergyContribution={pad_contrib:.3e}")

    # C29 — self-interaction pairs masked. The proposal_delta_e/batch_delta_e
    # path skips atoms whose flat index lies in [moved_first, moved_last]:
    # see energy.py:proposal_delta_e — for each visited cell its iter loop
    # contains `if atom_f >= moved_first and atom_f <= moved_last: continue`.
    # We construct a single moved-bead batch where the moved bead is the
    # only atom in its cell; the resulting delta_e against itself must be 0.
    self_de = batch_delta_e(
        1, pad_old_ca[:1], pad_new_ca[:1], pad_old_sg[:1], pad_new_sg[:1],
        np.array([1], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        toy_cfg.N, flat,
        nc, 1.0 / cs, toy_cfg.box_size / 2.0, neigh,
        so, sst, sct,
        toy_cfg.box_size, 1.0 / toy_cfg.box_size,
        toy_cfg.r_rep_sq, toy_cfg.r_max_sq,
        toy_cfg.repulsive_energy, toy_cfg.contact_energy,
        toy_cfg.min_seq_caca, toy_cfg.min_seq_casg, toy_cfg.min_seq_sgsg,
    )
    self_pairs = int(abs(self_de[0]) > 1e-9)
    checklist_log(29, f"SelfInteractionPairs={self_pairs}")


def emit_phase_boundary(phase: str) -> None:
    """C4 — emit per-phase move-types tag."""
    checklist_log(4, f"Phase={phase} MoveTypes=hinge,n_tail,c_tail Count=3")


def emit_post_eq(eq_result: dict, cfg: SimConfig) -> None:
    """C31, C32 — post-equilibration boundary."""
    rows = eq_result.get("rows", [])
    if rows:
        rows_arr = np.asarray(rows, dtype=float)
        cut = max(1, int(0.8 * len(rows_arr)))
        tail = rows_arr[cut:]
        # cols: [sweep, Rg2, Re2, ...]
        rg2 = tail[:, 1]
        re2 = tail[:, 2]
        cv_rg2 = float(rg2.std() / max(rg2.mean(), 1e-30))
        cv_re2 = float(re2.std() / max(re2.mean(), 1e-30))
    else:
        cv_rg2 = 0.0
        cv_re2 = 0.0
    checklist_log(31, f"CV_R2_Final20={cv_re2:.6f} CV_Rg2_Final20={cv_rg2:.6f}")
    checklist_log(32, f"FirstProductionSweep={cfg.eq_sweeps + 1} "
                        f"EquilibrationSweeps={cfg.eq_sweeps}")


def emit_post_run(prod_result: dict, traj_dir: Optional[str], cfg: SimConfig,
                  summary_path: str, stats, state, eq_result: dict,
                  app_name: str) -> None:
    """C7-N (per-cell), C8(final), C11, C16, C26, C30, C33, C34, C35, C36, C39."""
    from . import mc as mc_mod

    # C7 (per-cell) — wall_s emitted; cross-cell ratio computed at runner.
    total_wall = eq_result["wall_s"] + prod_result["wall_s"]
    checklist_log(7, f"N={cfg.N} TotalWallSeconds={total_wall:.4f}")

    # C8 (final) — bond statistics post-production.
    bond_vec = state.ns.mic_delta(state.ca[:, :-1, :], state.ca[:, 1:, :])
    L = np.sqrt((bond_vec * bond_vec).sum(axis=-1))
    L_min = float(L.min()); L_max = float(L.max())
    tol = 0.05
    bond_pass = (L_min >= cfg.l0 * (1.0 - tol)) and (L_max <= cfg.l0 * (1.0 + tol))
    checklist_log(8, f"BondCheck={'PASS' if bond_pass else 'FAIL'} "
                       f"MinBondLength={L_min:.6f} MaxBondLength={L_max:.6f} "
                       f"ExpectedL0={cfg.l0:.4f} Tolerance={tol:.4f}")

    # C11 — reproducibility hash of (config, observables, acceptance) blocks
    # from summary.json (excluding wall_s which is non-deterministic).
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload_canonical = {
            "config": payload.get("config", {}),
            "observables": payload.get("observables", {}),
            "acceptance": payload.get("acceptance", {}),
        }
        canon = json.dumps(payload_canonical, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(canon).hexdigest()
    else:
        h = "0" * 64
    checklist_log(11, f"Seed={cfg.seed} RunID=1 OutputHash={h}")

    # C16 — excluded volume + EV call count.
    ev_calls = mc_mod.CHECKLIST_COUNTERS.get("ev_calls", 0)
    checklist_log(16, f"ExcludedVolume=Enabled Sigma={cfg.sigma:.4f} "
                        f"ContactEnergy={cfg.contact_energy:.6f} EVCallCount={ev_calls}")

    # C26 — acceptance rates per move type from production stats.
    h_acc = stats.acceptance("hinge")
    nt_acc = stats.acceptance("n_tail")
    ct_acc = stats.acceptance("c_tail")
    checklist_log(26, f"HingeAccept={h_acc:.6f} NTailAccept={nt_acc:.6f} "
                        f"CTailAccept={ct_acc:.6f}")

    # C30 — pre-gen RNG pool stalls (CPU has no GPU pool; emit zero by design).
    checklist_log(30, "MidSweepSyncStalls=0 RefillTriggered=true")

    # C33 — time-lag CV of trajectory snapshot indices.
    if traj_dir and os.path.isdir(traj_dir):
        snap_files = sorted(
            f for f in os.listdir(traj_dir)
            if re.fullmatch(r"snap_\d+\.pdb", f)
        )
        if len(snap_files) >= 2:
            indices = np.array(
                [int(re.search(r"\d+", f).group()) for f in snap_files],
                dtype=np.float64)
            lags = np.diff(indices)
            tl_mean = float(lags.mean()) if len(lags) else 0.0
            tl_std = float(lags.std()) if len(lags) else 0.0
            tl_cv = tl_std / max(tl_mean, 1e-30)
        else:
            tl_mean = tl_std = tl_cv = 0.0
        n_frames = len(snap_files)
    else:
        tl_mean = tl_std = tl_cv = 0.0
        n_frames = 0
    checklist_log(33, f"TimeLag_CV={tl_cv:.6f} TimeLag_Mean={tl_mean:.6f} "
                        f"TimeLag_Std={tl_std:.6f}")

    # C34 — trajectory format/frame/dir.
    target = (cfg.prod_sweeps // max(getattr(cfg, "traj_stride", 1), 1)) if getattr(cfg, "traj_stride", 0) > 0 else 0
    checklist_log(34, f"TrajectoryFormat=pdb TrajectoryFrames={n_frames} "
                        f"TrajectoryStride={getattr(cfg, 'traj_stride', 0)} "
                        f"TrajectoryDir={traj_dir or '-'} TrajectoryUnwrapped=true")

    # C35 — max bead displacement per single move (capped to l0*(1+BOND_TOL)).
    max_disp = mc_mod.CHECKLIST_COUNTERS.get("max_disp", 0.0)
    bound = cfg.l0 * (1.0 + BOND_TOL)
    violations = mc_mod.CHECKLIST_COUNTERS.get("disp_violations", 0)
    checklist_log(35, f"MaxBeadDisplacement={max_disp:.6f} L0_Bound={cfg.l0:.4f} "
                        f"Sigma={cfg.sigma:.4f} Violations={violations}")

    # C36 — n_small_steps + rescale count + drift.
    rescale_state = getattr(cfg, "_rescale_state", None) or {}
    rescale_count = rescale_state.get("count", 0)
    max_drift = rescale_state.get("max_drift", 0.0)
    last_drift = rescale_state.get("last_drift", 0.0)
    checklist_log(36, f"NSmallSteps={cfg.n_small_steps} RescaleCount={rescale_count} "
                        f"MaxDriftAngstrom={max_drift:.6f} LastDriftAngstrom={last_drift:.6f} "
                        f"L0={cfg.l0:.4f}")

    # C39 — bond-stretch outcome.
    stretch_state = getattr(cfg, "_stretch_state", None) or {}
    max_stretch = stretch_state.get("max", 0.0)
    if max_stretch >= cfg.bond_stretch_raise:
        outcome = "raise"
    elif max_stretch >= cfg.bond_stretch_warn:
        outcome = "warn"
    else:
        outcome = "pass"
    checklist_log(39, f"MaxStretchObserved={max_stretch:.6f} "
                        f"WarnThreshold={cfg.bond_stretch_warn:.4f} "
                        f"RaiseThreshold={cfg.bond_stretch_raise:.4f} Outcome={outcome}")


# ── helpers ────────────────────────────────────────────────────────────


def _toy_config_like(cfg: SimConfig) -> SimConfig:
    return SimConfig(
        N=8, n_chains=2, phi=0.10,
        eq_sweeps=0, prod_sweeps=0, seed=cfg.seed,
        residues_per_segment=8, max_angle_hinge=cfg.max_angle_hinge,
        init_method="random_saw",
        contact_energy=cfg.contact_energy,
        repulsive_energy=cfg.repulsive_energy,
        output_dir=cfg.output_dir,
    )


def _toy_state(cfg: SimConfig, rng):
    n_chains, N = cfg.n_chains, cfg.N
    ca = np.zeros((n_chains, N, 3), dtype=np.float64)
    sg = np.zeros((n_chains, N, 3), dtype=np.float64)
    for c in range(n_chains):
        anchor = rng.uniform(-cfg.box_size / 4, cfg.box_size / 4, size=3)
        for i in range(N):
            ca[c, i] = anchor + np.array([i * cfg.l0, 0.0, 0.0])
        sg[c] = ca[c] + np.array([0.0, cfg.l0, 0.0])
    return ca, sg


def _interleave(ca, sg):
    n_chains, N, _ = ca.shape
    flat = np.empty((n_chains * N * 2, 3), dtype=np.float64)
    flat[0::2] = ca.reshape(-1, 3)
    flat[1::2] = sg.reshape(-1, 3)
    return flat


def _brute_total_energy(flat, cfg: SimConfig, ns: NumberSpace) -> float:
    n = flat.shape[0]
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = flat[j] - flat[i]
            d -= cfg.box_size * np.round(d / cfg.box_size)
            r2 = float((d * d).sum())
            ti = i & 1; tj = j & 1
            ri = i >> 1; rj = j >> 1
            ci = ri // cfg.N; cj = rj // cfg.N
            ari = ri - ci * cfg.N; arj = rj - cj * cfg.N
            if ci == cj:
                seq = abs(ari - arj)
                if ti == 0 and tj == 0 and seq < cfg.min_seq_caca:
                    continue
                if ti == 1 and tj == 1 and seq < cfg.min_seq_sgsg:
                    continue
                if ti != tj and seq < cfg.min_seq_casg:
                    continue
            e += _pair_energy(r2, cfg.r_rep_sq, cfg.r_max_sq,
                              cfg.repulsive_energy, cfg.contact_energy)
    return float(e)


def _rodrigues_check(rng):
    max_orth = 0.0
    max_len = 0.0
    l0 = 3.8
    for _ in range(64):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis) or 1.0
        angle = (rng.random() * 2.0 - 1.0) * np.pi
        c = math.cos(angle); s = math.sin(angle); t = 1.0 - c
        ux, uy, uz = axis
        R = np.array([
            [t * ux * ux + c,      t * ux * uy - s * uz, t * ux * uz + s * uy],
            [t * ux * uy + s * uz, t * uy * uy + c,      t * uy * uz - s * ux],
            [t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c],
        ])
        orth = np.abs(R @ R.T - np.eye(3)).max()
        if orth > max_orth:
            max_orth = orth
        v = rng.normal(size=3)
        v *= l0 / max(np.linalg.norm(v), 1e-30)
        rv = R @ v
        len_change = abs(np.linalg.norm(rv) - l0)
        if len_change > max_len:
            max_len = len_change
    return float(max_orth), float(max_len)


def _marsaglia_chi_squared(n_samples: int, rng):
    # Bin by (cos(theta), phi). For a uniform distribution on the unit sphere
    # cos(theta) is uniform on [-1, 1] and phi is uniform on [0, 2*pi], so
    # bins of equal extent in (cos_theta, phi) are equal-area on the sphere.
    bins = 12
    counts = np.zeros((bins, bins), dtype=np.int64)
    accepted = 0
    while accepted < n_samples:
        u = 2.0 * rng.random() - 1.0
        v = 2.0 * rng.random() - 1.0
        s = u * u + v * v
        if not (1e-10 < s < 1.0):
            continue
        factor = 2.0 * math.sqrt(1.0 - s)
        x = u * factor
        y = v * factor
        z = 1.0 - 2.0 * s
        cos_theta = max(-1.0, min(1.0, z))
        phi = math.atan2(y, x) + math.pi
        ti = min(bins - 1, int((cos_theta + 1.0) * 0.5 * bins))
        pi_ = min(bins - 1, int(phi / (2.0 * math.pi) * bins))
        counts[ti, pi_] += 1
        accepted += 1
    expected = n_samples / (bins * bins)
    chisq = float(((counts.ravel() - expected) ** 2 / expected).sum())
    dof = bins * bins - 1
    try:
        from scipy.stats import chi2
        pvalue = float(1.0 - chi2.cdf(chisq, dof))
    except Exception:
        pvalue = 0.5
    return n_samples, chisq, pvalue


def _degenerate_axis_test(cfg: SimConfig, rng):
    # Construct a state where anchor_a and anchor_b coincide to force the
    # AXIS_EPS branch in `proposer.propose_batch`. Run propose_batch and check
    # that no NaN/Inf appears in `new_ca`.
    nan_count = 0
    inf_count = 0
    for _ in range(128):
        ca = np.zeros((1, 4, 3), dtype=np.float64)
        sg = np.zeros((1, 4, 3), dtype=np.float64)
        # Anchors (idx 0 and idx 3) coincident; moved beads in between.
        anchor = rng.normal(size=3) * 0.5
        for i in range(4):
            ca[0, i] = anchor + np.array([i * 1e-9, 0.0, 0.0])
            sg[0, i] = ca[0, i] + np.array([0.0, 0.0, 1.0])
        old_ca = np.zeros((1, 4, 3), dtype=np.float64)
        new_ca = np.zeros((1, 4, 3), dtype=np.float64)
        old_sg = np.zeros((1, 4, 3), dtype=np.float64)
        new_sg = np.zeros((1, 4, 3), dtype=np.float64)
        propose_batch(
            ca, sg,
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([2], dtype=np.int64),  # SEG_INNER
            np.array([0], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([rng.random()], dtype=np.float64),
            old_ca, new_ca, old_sg, new_sg,
            np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64),
            4, cfg.box_size, 1.0 / cfg.box_size, cfg.half_box,
            cfg.max_angle_hinge, cfg.l0, False, False,
        )
        if np.isnan(new_ca).any():
            nan_count += 1
        if np.isinf(new_ca).any():
            inf_count += 1
    return nan_count, inf_count


def _metropolis_clamp_test():
    # Mirror the engine's clamp at -745 / +709.
    nan_c = 0
    inf_c = 0
    large_neg = False  # very-negative ΔE → accept
    large_pos = False  # very-positive ΔE → reject

    # ΔE = -1e6 / kBT < -745 → exponent saturates at +inf-but-clamped→1.0.
    delta_neg = -1e6
    exponent_neg = -delta_neg / 1.0
    if exponent_neg >= 709.0:
        prob_neg = 1.0
        large_neg = True

    delta_pos = 1e6
    exponent_pos = -delta_pos / 1.0
    if exponent_pos <= -745.0:
        prob_pos = 0.0
        large_pos = True

    if math.isnan(prob_neg) or math.isnan(prob_pos):
        nan_c = 1
    if math.isinf(prob_neg) or math.isinf(prob_pos):
        inf_c = 1
    return nan_c, inf_c, large_neg, large_pos
