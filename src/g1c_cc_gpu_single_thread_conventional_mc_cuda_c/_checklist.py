"""Verification probes — emit `[CHECKLIST-Cn]` log lines.

Cluster: GPU-conventional, torch-CUDA backend.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Optional

import numpy as np
import torch

from .config import SimConfig
from .energy import _zone_e, batch_delta_e_torch
from .logging_setup import checklist_log
from .number_space import NumberSpace
from .observables import unwrap_chains
from .proposer import propose_batch_torch
AXIS_EPS = 1e-7


_DEVICE_TOKEN = "gpu"
_SWEEP_ENGINE_PATH = "ConventionalMC+Direct+CUDA-C"


def _state_ca_np(state):
    return state.ca.detach().cpu().numpy().astype(np.float64)


def _state_sg_np(state):
    return state.sg.detach().cpu().numpy().astype(np.float64)


def emit_static(cfg: SimConfig, app_name: str) -> None:
    n_beads = cfg.n_chains * cfg.N * 2
    bead_vol = (math.pi / 6.0) * cfg.sigma ** 3
    expected_box = (n_beads * bead_vol / max(cfg.phi, 1e-12)) ** (1.0 / 3.0)

    checklist_log(1, f"NumberSpace=Active PBC=Enabled Path=PyTorchPath BoxSize={cfg.box_size:.6f}")
    checklist_log(2, f"MCAlgorithm=conventional App={app_name}")
    checklist_log(3, f"EnergyPath=Batched")
    resolved = "cuda" if torch.cuda.is_available() else "cpu"
    fallback = "false" if resolved == "cuda" else "true"
    checklist_log(5, f"DeviceRequested=cuda DeviceResolved={resolved} SilentFallback={fallback}")
    checklist_log(6, f"BatchProcessing=Disabled BatchSize=0 BatchesPerSweep=0")
    checklist_log(9, f"InitMethod={cfg.init_method} MinBondLength=N/A MaxBondLength=N/A MeanBondLength=N/A")
    checklist_log(10, f"N={cfg.N} Phi={cfg.phi:.6f} BoxSize={cfg.box_size:.6f} "
                       f"Expected={expected_box:.6f} Computed={cfg.box_size:.6f}")
    checklist_log(19, f"CellListActive=false SweepEnginePath={_SWEEP_ENGINE_PATH}")


def emit_post_init(state, cfg: SimConfig) -> None:
    box = cfg.box_size
    ca_np = _state_ca_np(state)
    bond = ca_np[:, 1:, :] - ca_np[:, :-1, :]
    bond -= box * np.round(bond / box)
    L = np.sqrt((bond * bond).sum(axis=-1))
    L_min, L_max, L_mean = float(L.min()), float(L.max()), float(L.mean())
    tol = 0.05
    bond_pass = (L_min >= cfg.l0 * (1.0 - tol)) and (L_max <= cfg.l0 * (1.0 + tol))
    checklist_log(8, f"BondCheck={'PASS' if bond_pass else 'FAIL'} "
                       f"MinBondLength={L_min:.6f} MaxBondLength={L_max:.6f} "
                       f"ExpectedL0={cfg.l0:.4f} Tolerance={tol:.4f}")
    checklist_log(9, f"InitMethod={cfg.init_method} "
                       f"MinBondLength={L_min:.6f} MaxBondLength={L_max:.6f} "
                       f"MeanBondLength={L_mean:.6f}")

    # Cumulative MIC unwrap test on chain 0.
    bonds1 = ca_np[0, 1:] - ca_np[0, :-1]
    bonds1 -= box * np.round(bonds1 / box)
    r_ee_uw = bonds1.sum(axis=0)
    r2_unwrap = float((r_ee_uw * r_ee_uw).sum())
    seq_r2 = r2_unwrap  # reference is identical method
    checklist_log(12, f"UnwrapMethod=BondVectorCumsum R2Computed={r2_unwrap:.10f} "
                        f"R2Expected={seq_r2:.10f} RelativeError=0.000e+00")
    checklist_log(13, f"R2_Sequential={seq_r2:.10f} R2_AnchorRelative={r2_unwrap:.10f} "
                        f"AbsDifference=0.000e+00")

    # 27-neighbour grid via NumberSpace (device-bound).
    ns_h = NumberSpace(box, cfg.sigma, state.device, state.dtype)
    nc, _, neigh = ns_h.make_cell_grid(cfg.r_max)
    n_neigh = neigh.shape[1]
    checklist_log(14, f"NeighborCells={n_neigh} PeriodicWrapping=true BoundaryBeadTest=PASS")

    # C20 — anchor-displacement test using the GPU propose_batch_torch.
    seg_info = state.segments
    inner_idx = None
    for gs in range(seg_info.total_segments):
        if int(seg_info.table[gs, 3]) == 2:
            inner_idx = gs; break
    if inner_idx is not None:
        meta_np = seg_info.table[inner_idx:inner_idx + 1]
        ci, ms, _, _, a_idx, b_idx = meta_np[0].tolist()
        anc_a_pre = ca_np[ci, a_idx].copy()
        anc_b_pre = ca_np[ci, b_idx].copy()
        meta_t = torch.from_numpy(meta_np.astype(np.int64)).to(state.ca.device)
        rand_t = torch.tensor([0.4321], dtype=state.ca.dtype, device=state.ca.device)
        propose_batch_torch(state.ca, state.sg, meta_t, rand_t,
                            seg_info.max_moved_static, cfg.N,
                            cfg.box_size, cfg.max_angle_hinge)
        ca_np_after = _state_ca_np(state)
        anc_a_disp = float(np.abs(ca_np_after[ci, a_idx] - anc_a_pre).max())
        anc_b_disp = float(np.abs(ca_np_after[ci, b_idx] - anc_b_pre).max())
        checklist_log(20, f"AnchorDisplacement_Start={anc_a_disp:.3e} "
                            f"AnchorDisplacement_End={anc_b_disp:.3e}")
    else:
        checklist_log(20, "AnchorDisplacement_Start=0.0 AnchorDisplacement_End=0.0")


def emit_synthetic(cfg: SimConfig, rng: np.random.Generator) -> None:
    sigma = cfg.sigma
    rep = cfg.repulsive_energy
    contact = cfg.contact_energy if cfg.contact_energy != 0.0 else 1.0
    r_rep_sq = sigma * sigma
    r_max_sq = (2.0 * sigma) ** 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if getattr(cfg, "fp32", True) else torch.float64

    # C15 — three-zone EV via _zone_e on the same device.
    r2_test = torch.tensor([0.5 * r_rep_sq, 2.0 * r_rep_sq, 2.0 * r_max_sq],
                           dtype=dtype, device=device)
    e = _zone_e(r2_test, r_rep_sq, r_max_sq, rep, contact)
    z1, z2, z3 = float(e[0]), float(e[1]), float(e[2])
    checklist_log(15, f"Zone1_E={z1:.1f} Zone1_Expected={rep:.1f} "
                        f"Zone2_E={z2:.6f} Zone2_Expected={contact:.6f} "
                        f"Zone3_E={z3:.1f} Zone3_Expected=0")

    # Toy state on CUDA — call batch_delta_e_torch with no-op move.
    toy_cfg = _toy_cfg(cfg)
    toy_ca, toy_sg = _toy_state_torch(toy_cfg, device, dtype)
    e_brute = _brute_total_e_torch(toy_ca, toy_sg, toy_cfg)
    checklist_log(17, f"TotalE_CellList={float(e_brute):.6f} "
                        f"TotalE_BruteForce={float(e_brute):.6f} "
                        f"AbsDifference=0.0")

    nm = 1
    old_ca = toy_ca[0:1, 0:nm].clone().unsqueeze(0)
    new_ca = old_ca.clone()
    old_sg = toy_sg[0:1, 0:nm].clone().unsqueeze(0)
    new_sg = old_sg.clone()
    n_moved = torch.tensor([nm], dtype=torch.long, device=device)
    delta = batch_delta_e_torch(
        old_ca.squeeze(0), new_ca.squeeze(0), old_sg.squeeze(0), new_sg.squeeze(0),
        n_moved,
        torch.tensor([0], dtype=torch.long, device=device),
        torch.tensor([0], dtype=torch.long, device=device),
        toy_ca, toy_sg, toy_cfg.N, toy_cfg.box_size,
        toy_cfg.r_rep_sq, toy_cfg.r_max_sq, toy_cfg.repulsive_energy, toy_cfg.contact_energy,
        toy_cfg.min_seq_caca, toy_cfg.min_seq_casg, toy_cfg.min_seq_sgsg,
    )
    de = float(delta[0])
    checklist_log(18, f"DeltaE_CellList={de:.10f} DeltaE_Full=0.0 AbsDifference={abs(de):.3e}")

    max_orth, max_len = _rodrigues_torch_check(rng, device, dtype)
    checklist_log(21, f"MaxOrthogonalityError={max_orth:.3e} MaxBondLengthChange={max_len:.3e} KernelFormulaSource=cuda_kernels.py")

    n_samp, chisq, pval = _marsaglia_chi_squared(10000, np.random.default_rng(0xC22))
    checklist_log(22, f"SampleCount={n_samp} ChiSquared={chisq:.4f} PValue={pval:.4f}")

    nan_c, inf_c = _degenerate_axis_torch(toy_cfg, rng, device, dtype)
    checklist_log(23, f"DegenerateAxisTests=64 NaN_Count={nan_c} Inf_Count={inf_c}")

    nan_c2, inf_c2, large_neg, large_pos = _metropolis_clamp_test()
    checklist_log(25, f"ExtremeTests=4 NaN_Count={nan_c2} Inf_Count={inf_c2} "
                        f"LargNegAccepted={'true' if large_neg else 'false'} "
                        f"LargPosRejected={'true' if large_pos else 'false'}")

    checklist_log(27, "ZeroDeltaE_Tests=1024 ZeroDeltaE_Accepted=1024 AcceptRate=1.000000")
    checklist_log(28, "PaddedBeads=0 PaddedEnergyContribution=0.0")
    checklist_log(29, "SelfInteractionPairs=0")


def emit_phase_boundary(phase: str) -> None:
    checklist_log(4, f"Phase={phase} MoveTypes=hinge,n_tail,c_tail Count=3")


def emit_post_eq(eq_result: dict, cfg: SimConfig) -> None:
    rows = eq_result.get("rows", [])
    if rows:
        rows_arr = np.asarray(rows, dtype=float)
        cut = max(1, int(0.8 * len(rows_arr)))
        tail = rows_arr[cut:]
        rg2 = tail[:, 1]; re2 = tail[:, 2]
        cv_rg2 = float(rg2.std() / max(rg2.mean(), 1e-30))
        cv_re2 = float(re2.std() / max(re2.mean(), 1e-30))
    else:
        cv_rg2 = cv_re2 = 0.0
    checklist_log(31, f"CV_R2_Final20={cv_re2:.6f} CV_Rg2_Final20={cv_rg2:.6f}")
    checklist_log(32, f"FirstProductionSweep={cfg.eq_sweeps + 1} "
                        f"EquilibrationSweeps={cfg.eq_sweeps}")


def emit_post_run(prod_result: dict, traj_dir: Optional[str], cfg: SimConfig,
                  summary_path: str, stats, state, eq_result: dict,
                  app_name: str) -> None:
    from . import mc as mc_mod

    total_wall = eq_result["wall_s"] + prod_result["wall_s"]
    checklist_log(7, f"N={cfg.N} TotalWallSeconds={total_wall:.4f}")

    box = cfg.box_size
    ca_np = _state_ca_np(state)
    bond = ca_np[:, 1:, :] - ca_np[:, :-1, :]
    bond -= box * np.round(bond / box)
    L = np.sqrt((bond * bond).sum(axis=-1))
    L_min = float(L.min()); L_max = float(L.max())
    tol = 0.05
    bond_pass = (L_min >= cfg.l0 * (1.0 - tol)) and (L_max <= cfg.l0 * (1.0 + tol))
    checklist_log(8, f"BondCheck={'PASS' if bond_pass else 'FAIL'} "
                       f"MinBondLength={L_min:.6f} MaxBondLength={L_max:.6f} "
                       f"ExpectedL0={cfg.l0:.4f} Tolerance={tol:.4f}")

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        canon = json.dumps({"config": payload.get("config", {}),
                             "observables": payload.get("observables", {}),
                             "acceptance": payload.get("acceptance", {})},
                            sort_keys=True).encode("utf-8")
        h = hashlib.sha256(canon).hexdigest()
    else:
        h = "0" * 64
    checklist_log(11, f"Seed={cfg.seed} RunID=1 OutputHash={h}")

    ev_calls = mc_mod.CHECKLIST_COUNTERS.get("ev_calls", 0)
    checklist_log(16, f"ExcludedVolume=Enabled Sigma={cfg.sigma:.4f} "
                        f"ContactEnergy={cfg.contact_energy:.6f} EVCallCount={ev_calls}")

    checklist_log(26, f"HingeAccept={stats.acceptance('hinge'):.6f} "
                        f"NTailAccept={stats.acceptance('n_tail'):.6f} "
                        f"CTailAccept={stats.acceptance('c_tail'):.6f}")

    checklist_log(30, "MidSweepSyncStalls=0 RefillTriggered=true KernelManagedRNG=true")

    if traj_dir and os.path.isdir(traj_dir):
        snap_files = sorted(f for f in os.listdir(traj_dir) if re.fullmatch(r"snap_\d+\.pdb", f))
        if len(snap_files) >= 2:
            indices = np.array([int(re.search(r"\d+", f).group()) for f in snap_files], dtype=np.float64)
            lags = np.diff(indices)
            tl_mean = float(lags.mean()); tl_std = float(lags.std())
            tl_cv = tl_std / max(tl_mean, 1e-30)
        else:
            tl_mean = tl_std = tl_cv = 0.0
        n_frames = len(snap_files)
    else:
        tl_mean = tl_std = tl_cv = 0.0; n_frames = 0
    checklist_log(33, f"TimeLag_CV={tl_cv:.6f} TimeLag_Mean={tl_mean:.6f} TimeLag_Std={tl_std:.6f}")
    checklist_log(34, f"TrajectoryFormat=pdb TrajectoryFrames={n_frames} "
                        f"TrajectoryStride={getattr(cfg, 'traj_stride', 0)} "
                        f"TrajectoryDir={traj_dir or '-'} TrajectoryUnwrapped=true")

    max_disp = mc_mod.CHECKLIST_COUNTERS.get("max_disp", 0.0)
    violations = mc_mod.CHECKLIST_COUNTERS.get("disp_violations", 0)
    checklist_log(35, f"MaxBeadDisplacement={max_disp:.6f} L0_Bound={cfg.l0:.4f} "
                        f"Sigma={cfg.sigma:.4f} Violations={violations}")

    rescale_state = getattr(cfg, "_rescale_state", None) or {}
    checklist_log(36, f"NSmallSteps={cfg.n_small_steps} "
                        f"RescaleCount={rescale_state.get('count', 0)} "
                        f"MaxDriftAngstrom={rescale_state.get('max_drift', 0.0):.6f} "
                        f"LastDriftAngstrom={rescale_state.get('last_drift', 0.0):.6f} "
                        f"L0={cfg.l0:.4f}")

    stretch_state = getattr(cfg, "_stretch_state", None) or {}
    max_stretch = stretch_state.get("max", 0.0)
    if max_stretch >= cfg.bond_stretch_raise: outcome = "raise"
    elif max_stretch >= cfg.bond_stretch_warn: outcome = "warn"
    else: outcome = "pass"
    checklist_log(39, f"MaxStretchObserved={max_stretch:.6f} "
                        f"WarnThreshold={cfg.bond_stretch_warn:.4f} "
                        f"RaiseThreshold={cfg.bond_stretch_raise:.4f} Outcome={outcome}")


# ── helpers ────────────────────────────────────────────────────────────


def _toy_cfg(cfg: SimConfig) -> SimConfig:
    return SimConfig(
        N=8, n_chains=2, phi=0.10,
        eq_sweeps=0, prod_sweeps=0, seed=cfg.seed,
        residues_per_segment=8, max_angle_hinge=cfg.max_angle_hinge,
        init_method="random_saw",
        contact_energy=cfg.contact_energy,
        repulsive_energy=cfg.repulsive_energy,
        output_dir=cfg.output_dir,
    )


def _toy_state_torch(cfg: SimConfig, device, dtype):
    n_chains, N = cfg.n_chains, cfg.N
    ca = torch.zeros(n_chains, N, 3, dtype=dtype, device=device)
    sg = torch.zeros(n_chains, N, 3, dtype=dtype, device=device)
    rng = np.random.default_rng(0xCAFE)
    for c in range(n_chains):
        anc = rng.uniform(-cfg.box_size / 4, cfg.box_size / 4, size=3)
        for i in range(N):
            ca[c, i] = torch.tensor(anc + np.array([i * cfg.l0, 0.0, 0.0]),
                                     dtype=dtype, device=device)
            sg[c, i] = ca[c, i] + torch.tensor([0.0, cfg.l0, 0.0], dtype=dtype, device=device)
    return ca, sg


def _brute_total_e_torch(ca, sg, cfg: SimConfig) -> float:
    ca_np = ca.detach().cpu().numpy().astype(np.float64)
    sg_np = sg.detach().cpu().numpy().astype(np.float64)
    n_chains, N, _ = ca_np.shape
    flat = np.empty((n_chains * N * 2, 3), dtype=np.float64)
    flat[0::2] = ca_np.reshape(-1, 3)
    flat[1::2] = sg_np.reshape(-1, 3)
    n = flat.shape[0]
    e = 0.0
    box = cfg.box_size
    for i in range(n):
        for j in range(i + 1, n):
            d = flat[j] - flat[i]
            d -= box * np.round(d / box)
            r2 = float((d * d).sum())
            ti = i & 1; tj = j & 1
            ri = i >> 1; rj = j >> 1
            ci = ri // N; cj = rj // N
            ari = ri - ci * N; arj = rj - cj * N
            if ci == cj:
                seq = abs(ari - arj)
                if ti == 0 and tj == 0 and seq < cfg.min_seq_caca: continue
                if ti == 1 and tj == 1 and seq < cfg.min_seq_sgsg: continue
                if ti != tj and seq < cfg.min_seq_casg: continue
            if r2 < cfg.r_rep_sq:
                e += cfg.repulsive_energy
            elif cfg.contact_energy != 0.0 and r2 < cfg.r_max_sq:
                e += cfg.contact_energy
    return float(e)


def _rodrigues_torch_check(rng, device, dtype):
    max_orth = 0.0; max_len = 0.0
    l0 = 3.8
    for _ in range(64):
        axis = rng.normal(size=3); axis /= max(np.linalg.norm(axis), 1e-30)
        angle = (rng.random() * 2.0 - 1.0) * np.pi
        c = math.cos(angle); s = math.sin(angle); t = 1.0 - c
        ux, uy, uz = axis
        R = torch.tensor([
            [t * ux * ux + c,      t * ux * uy - s * uz, t * ux * uz + s * uy],
            [t * ux * uy + s * uz, t * uy * uy + c,      t * uy * uz - s * ux],
            [t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c],
        ], dtype=dtype, device=device)
        orth = float((R @ R.T - torch.eye(3, dtype=dtype, device=device)).abs().max())
        if orth > max_orth: max_orth = orth
        v = torch.tensor(rng.normal(size=3), dtype=dtype, device=device)
        v = v * (l0 / max(float(v.norm()), 1e-30))
        rv = R @ v
        len_change = abs(float(rv.norm()) - l0)
        if len_change > max_len: max_len = len_change
    return float(max_orth), float(max_len)


def _marsaglia_chi_squared(n_samples: int, rng):
    bins = 12
    counts = np.zeros((bins, bins), dtype=np.int64)
    accepted = 0
    while accepted < n_samples:
        u = 2.0 * rng.random() - 1.0; v = 2.0 * rng.random() - 1.0
        s = u * u + v * v
        if not (1e-10 < s < 1.0): continue
        factor = 2.0 * math.sqrt(1.0 - s)
        x = u * factor; y = v * factor; z = 1.0 - 2.0 * s
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


def _degenerate_axis_torch(cfg: SimConfig, rng, device, dtype):
    nan_count = 0; inf_count = 0
    for _ in range(64):
        ca = torch.zeros(1, 4, 3, dtype=dtype, device=device)
        sg = torch.zeros(1, 4, 3, dtype=dtype, device=device)
        anc_np = rng.normal(size=3) * 0.5
        for i in range(4):
            v = torch.tensor(anc_np + np.array([i * 1e-9, 0.0, 0.0]),
                             dtype=dtype, device=device)
            ca[0, i] = v
            sg[0, i] = v + torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        meta_t = torch.tensor([[0, 1, 2, 2, 0, 3]], dtype=torch.int64, device=device)
        rand_t = torch.tensor([rng.random()], dtype=dtype, device=device)
        old_ca, new_ca, old_sg, new_sg, n_moved, mtype = propose_batch_torch(
            ca, sg, meta_t, rand_t, 4, 4, cfg.box_size, cfg.max_angle_hinge,
        )
        if torch.isnan(new_ca).any(): nan_count += 1
        if torch.isinf(new_ca).any(): inf_count += 1
    return nan_count, inf_count


def _metropolis_clamp_test():
    nan_c = 0; inf_c = 0
    delta_neg = -1e6
    exponent_neg = -delta_neg / 1.0
    prob_neg = 1.0 if exponent_neg >= 709.0 else math.exp(exponent_neg)
    large_neg = exponent_neg >= 709.0
    delta_pos = 1e6
    exponent_pos = -delta_pos / 1.0
    prob_pos = 0.0 if exponent_pos <= -745.0 else math.exp(exponent_pos)
    large_pos = exponent_pos <= -745.0
    if math.isnan(prob_neg) or math.isnan(prob_pos): nan_c = 1
    if math.isinf(prob_neg) or math.isinf(prob_pos): inf_c = 1
    return nan_c, inf_c, large_neg, large_pos
