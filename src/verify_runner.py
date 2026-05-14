"""verify_runner — scaled 576-cell verification matrix driver.

For every (app, N, phi, seed) cell, subprocess `python -m <app>.main`
with the cell's parameters; collect run.log + summary.json + observables.

After all cells:
  - Aggregate [CHECKLIST-Cn] log lines into c_results.tsv.
  - Compute T1..T7 fits per (app, phi) into t_results.tsv (numpy.polyfit
    on log-log of mean_R2/Rg2/tau_R/D vs N; ±15% tolerance, R² > 0.90).
  - Compute C7 wall_ratio per (app, phi) from N=25 / N=100 wall_s.
  - Render the 9 PNGs declared in §A3.plots.

Usage:
    python src/verify_runner.py --scaled --output_dir _verif_full
    python src/verify_runner.py --scaled --apps c1c_nb_cpu_single_core_conventional_mc_numba

Scaled params: n_chains=10, eq_sweeps=200, prod_sweeps=500, traj_stride=10,
caps enabled (--cap_inner_hinge --cap_tail).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Per context_rouse_verification.txt §8 — spec-conformant (NOT scaled).
# Sweep schedule per N: ≥10*tau_R equilibration, ≥30 decorrelation lengths in
# production. n_chains=100 keeps MC noise within ±5% Kuriata bounds.
SPEC = {
    "n_chains": 100,
    "schedule": {
        25:  dict(eq_sweeps=10000, prod_sweeps=37500),
        50:  dict(eq_sweeps=25000, prod_sweeps=75000),
        100: dict(eq_sweeps=50000, prod_sweeps=187500),
    },
    "batch_size_multistep": 100,
}
# Legacy scaled-mode dict kept for any caller still importing SCALED.
SCALED = dict(n_chains=10, eq_sweeps=200, prod_sweeps=500, traj_stride=10)
N_VALUES = [25, 50, 100]
PHI_VALUES = [0.03, 0.10, 0.20, 0.30]
SEEDS = [42, 43, 44]

T_TOLERANCE = 0.15  # ±15% Kuriata per protocol §8 (n_chains=10 scaled budget)
T_MIN_R2 = 0.90


def per_n_schedule(N: int) -> Dict[str, int]:
    s = SPEC["schedule"][N]
    eq = s["eq_sweeps"]; prod = s["prod_sweeps"]
    return {"eq_sweeps": eq, "prod_sweeps": prod,
            "traj_stride": max(1, (eq + prod) // 100)}

KURIATA_TARGETS = {
    "T1": (1.18, "log<R^2> vs log(N)"),
    "T2": (1.18, "log<Rg^2> vs log(N)"),
    "T3": (6.25, "<R^2>/<Rg^2> @ large N"),
    "T4": (1.0, "log gCM(t) vs log(t)"),
    "T5": (0.5, "log g1(t) vs log(t)"),
    "T6": (2.18, "log tau_R vs log(N)"),
    "T7": (-1.0, "log D vs log(N)"),
}


APPS_ALL = [
    "c1c_nb_cpu_single_core_conventional_mc_numba",
    "c1c_pt_cpu_single_core_conventional_mc_py_torch",
    "c1m_nb_cpu_single_core_multistep_mc_numba",
    "c1m_pt_cpu_single_core_multistep_mc_py_torch",
    "cnc_nb_cpu_multi_core_conventional_mc_numba",
    "cnc_pt_cpu_multi_core_conventional_mc_py_torch",
    "cnm_nb_cpu_multi_core_multistep_mc_numba",
    "cnm_pt_cpu_multi_core_multistep_mc_py_torch",
    "g1c_cc_gpu_single_thread_conventional_mc_cuda_c",
    "g1c_pt_gpu_single_thread_conventional_mc_py_torch",
    "g1m_cc_gpu_single_thread_multistep_mc_cuda_c",
    "g1m_pt_gpu_single_thread_multistep_mc_py_torch",
    "gnc_cc_gpu_multi_thread_conventional_mc_cuda_c",
    "gnc_pt_gpu_multi_thread_conventional_mc_py_torch",
    "gnm_cc_gpu_multi_thread_multistep_mc_cuda_c",
    "gnm_pt_gpu_multi_thread_multistep_mc_py_torch",
]


def is_gpu(app: str) -> bool:
    return app.startswith("gpu_")


def is_multistep(app: str) -> bool:
    return "multistep" in app


def is_multi_core(app: str) -> bool:
    return "multi_core" in app


def is_multi_thread(app: str) -> bool:
    return "multi_thread" in app


def needs_workers(app: str) -> bool:
    return app == "cnm_pt_cpu_multi_core_multistep_mc_py_torch"


def cell_dir(verif_root: Path, app: str, N: int, phi: float, seed: int) -> Path:
    return verif_root / app / f"N{N}_phi{phi:.2f}_seed{seed}"


def build_cmd(python_exe: str, app: str, N: int, phi: float, seed: int,
              cell_out: Path, scaled: bool = False) -> List[str]:
    if scaled:
        n_chains = SCALED["n_chains"]
        eq = SCALED["eq_sweeps"]; prod = SCALED["prod_sweeps"]
        stride = SCALED["traj_stride"]
        bs_multistep = 1
    else:
        n_chains = SPEC["n_chains"]
        sched = per_n_schedule(N)
        eq = sched["eq_sweeps"]; prod = sched["prod_sweeps"]
        stride = sched["traj_stride"]
        bs_multistep = SPEC["batch_size_multistep"]
    cmd = [
        python_exe, "-m", f"{app}.main",
        "--N", str(N),
        "--n_chains", str(n_chains),
        "--phi", f"{phi:.6f}",
        "--eq_sweeps", str(eq),
        "--prod_sweeps", str(prod),
        "--seed", str(seed),
        "--init_method", "random_saw",
        "--residues_per_segment", "8",
        "--max_angle_hinge_pi", "0.5",
        "--traj_stride", str(stride),
        "--contact_energy", "0.0",
        "--repulsive_energy", "1e6",
        "--output_dir", str(cell_out),
        "--log_level", "INFO",
        "--log_file", str(cell_out / "run.log"),
        "--cap_inner_hinge",
        "--cap_tail",
    ]
    if is_multistep(app):
        cmd += ["--batch_size", str(bs_multistep)]
    if is_multi_core(app):
        cmd += ["--threads", "1"]
    if needs_workers(app):
        cmd += ["--workers", "1"]
    if is_multi_thread(app):
        cmd += ["--streams", "1"]
    return cmd


def run_cell(args: Tuple) -> Dict:
    # 7-tuple: (app, N, phi, seed, verif_root, python_exe, scaled)
    app, N, phi, seed, verif_root, python_exe, scaled = args
    cell_out = cell_dir(verif_root, app, N, phi, seed)
    cell_out.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).resolve().parent
    cmd = build_cmd(python_exe, app, N, phi, seed, cell_out, scaled=scaled)
    t0 = time.time()
    try:
        cp = subprocess.run(
            cmd, cwd=str(src_dir),
            capture_output=True, text=True, timeout=14400,
        )
        wall = time.time() - t0
        return {
            "app": app, "N": N, "phi": phi, "seed": seed,
            "rc": cp.returncode, "wall_s": round(wall, 2),
            "cell_dir": str(cell_out),
            "stderr_tail": (cp.stderr or "").strip().splitlines()[-3:],
        }
    except subprocess.TimeoutExpired:
        return {
            "app": app, "N": N, "phi": phi, "seed": seed,
            "rc": -1, "wall_s": round(time.time() - t0, 2),
            "cell_dir": str(cell_out),
            "stderr_tail": ["timeout 14400s"],
        }


def parse_run_log(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    pat = re.compile(r"\[CHECKLIST-(C\d+|T\d+)\]\s+(.+)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_acceptance(items: Dict[str, str]) -> Dict[str, float]:
    line = items.get("C26", "")
    out = {"hinge": 0.0, "n_tail": 0.0, "c_tail": 0.0}
    for m in re.finditer(r"(HingeAccept|NTailAccept|CTailAccept)=([0-9.eE+-]+)", line):
        key = {"HingeAccept": "hinge", "NTailAccept": "n_tail",
               "CTailAccept": "c_tail"}[m.group(1)]
        out[key] = float(m.group(2))
    return out


def write_c_results(rows: List[Dict], verif_root: Path) -> None:
    path = verif_root / "c_results.tsv"
    fieldnames = ["app", "N", "phi", "seed", "c_item", "raw"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def aggregate_c_results(verif_root: Path, cells: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for cell in cells:
        if cell["rc"] != 0:
            continue
        d = Path(cell["cell_dir"])
        items = parse_run_log(d / "run.log")
        for c_item, raw in items.items():
            if not c_item.startswith("C"):
                continue
            rows.append({
                "app": cell["app"], "N": cell["N"], "phi": cell["phi"],
                "seed": cell["seed"], "c_item": c_item, "raw": raw,
            })
    return rows


def cell_observables(verif_root: Path, cell: Dict) -> Dict:
    """Read summary.json + run.log → derived metrics for T-item fits."""
    d = Path(cell["cell_dir"])
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    obs = summary.get("observables", {})
    timing = summary.get("timing", {})
    out = {
        "app": cell["app"], "N": cell["N"], "phi": cell["phi"], "seed": cell["seed"],
        "mean_R2": obs.get("re2_mean", float("nan")),
        "std_R2": obs.get("re2_std", float("nan")),
        "mean_Rg2": obs.get("rg2_mean", float("nan")),
        "std_Rg2": obs.get("rg2_std", float("nan")),
        "wall_eq": timing.get("eq_wall_s", float("nan")),
        "wall_prod": timing.get("prod_wall_s", float("nan")),
    }
    out["tau_R"], out["D"], out["gCM_slope"], out["g1_short_slope"] = \
        compute_dynamics(d, cell["N"])
    return out


def compute_dynamics(cell_dir_path: Path, N: int) -> Tuple[float, float, float, float]:
    """Compute (tau_R, D, gCM_loglog_slope, g1_short_loglog_slope) from PDB
    trajectory. tau_R = ACF 1/e crossing of R_ee. D = linear slope(gCM)/6.
    gCM_slope = log-log slope of gCM(t) (T4 target=1.0). g1_slope = log-log
    slope of g1(t) on short-time portion (T5 target=0.5)."""
    nan4 = (float("nan"),) * 4
    traj = cell_dir_path / "trajectory"
    if not traj.is_dir():
        return nan4
    snaps = sorted(traj.glob("snap_*.pdb"))
    if len(snaps) < 5:
        return nan4
    chains_per_frame: List[np.ndarray] = []
    for p in snaps:
        ca = read_pdb_ca(p, N)
        if ca is None:
            return nan4
        chains_per_frame.append(ca)
    arr = np.stack(chains_per_frame, axis=0)  # [T, n_chains, N, 3]
    T = arr.shape[0]
    R_ee = arr[:, :, -1, :] - arr[:, :, 0, :]
    R_cm = arr.mean(axis=2)
    mid = N // 2
    r_mid = arr[:, :, mid, :]

    R_ee0 = R_ee[0]
    norm = (R_ee0 * R_ee0).sum(axis=-1).mean()
    acf = np.array([(R_ee[t] * R_ee0).sum(axis=-1).mean() / max(norm, 1e-30)
                    for t in range(T)])
    tau_idx = np.argmax(acf < 1.0 / math.e) if (acf < 1.0 / math.e).any() else T - 1
    tau_R = float(tau_idx) if tau_idx > 0 else float("nan")

    gCM = np.array([(((R_cm[t] - R_cm[0]) ** 2).sum(axis=-1)).mean() for t in range(T)])
    g1 = np.array([(((r_mid[t] - r_mid[0]) ** 2).sum(axis=-1)).mean() for t in range(T)])

    lo = max(1, int(0.05 * T)); hi = max(lo + 2, int(0.7 * T))
    if hi - lo < 3:
        return tau_R, float("nan"), float("nan"), float("nan")
    t_axis = np.arange(lo, hi, dtype=np.float64)
    slope_lin, _ = np.polyfit(t_axis, gCM[lo:hi], 1)
    D = float(slope_lin / 6.0) if slope_lin > 0 else float("nan")

    # T4: log-log slope of gCM(t) over linear regime.
    def _loglog_slope(y: np.ndarray, lo: int, hi: int) -> float:
        seg = y[lo:hi]
        ts = t_axis
        mask = np.isfinite(seg) & (seg > 0)
        if mask.sum() < 3:
            return float("nan")
        s, _ = np.polyfit(np.log(ts[mask]), np.log(seg[mask]), 1)
        return float(s)
    gCM_loglog = _loglog_slope(gCM, lo, hi)

    # T5: g1 short-time log-log slope (frames 1..min(T/4, 30)).
    short_hi = max(4, min(T // 4, 30))
    short_lo = 1
    seg = g1[short_lo:short_hi]
    ts_short = np.arange(short_lo, short_hi, dtype=np.float64)
    mask = np.isfinite(seg) & (seg > 0)
    if mask.sum() >= 3:
        g1_short_slope, _ = np.polyfit(np.log(ts_short[mask]), np.log(seg[mask]), 1)
        g1_short_slope = float(g1_short_slope)
    else:
        g1_short_slope = float("nan")
    return tau_R, D, gCM_loglog, g1_short_slope


def read_pdb_ca(path: Path, N: int) -> Optional[np.ndarray]:
    """Return [n_chains, N, 3] of CA positions; None on parse error."""
    try:
        coords: List[Tuple[float, float, float]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("ATOM") and " CA " in line:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                coords.append((x, y, z))
        n_total = len(coords)
        if n_total == 0 or n_total % N != 0:
            return None
        n_chains = n_total // N
        return np.asarray(coords, dtype=np.float64).reshape(n_chains, N, 3)
    except Exception:
        return None


def fit_t_items(obs_rows: List[Dict]) -> List[Dict]:
    """Compute T1..T7 fits per (app, phi). T-rows: app, phi, T_item,
    fitted_value, regression_R2, in_range."""
    by_app_phi: Dict[Tuple[str, float], Dict[int, List[Dict]]] = {}
    for r in obs_rows:
        key = (r["app"], r["phi"])
        by_app_phi.setdefault(key, {}).setdefault(r["N"], []).append(r)

    out: List[Dict] = []
    for (app, phi), per_N in by_app_phi.items():
        # Median across seeds per N.
        vals_R2 = []; vals_Rg2 = []; vals_tau = []; vals_D = []
        Ns = []
        for N in sorted(per_N.keys()):
            rows = per_N[N]
            vals_R2.append(np.median([r["mean_R2"] for r in rows]))
            vals_Rg2.append(np.median([r["mean_Rg2"] for r in rows]))
            vals_tau.append(np.median([r["tau_R"] for r in rows
                                        if math.isfinite(r["tau_R"])]) if any(
                math.isfinite(r["tau_R"]) for r in rows) else float("nan"))
            vals_D.append(np.median([r["D"] for r in rows
                                      if math.isfinite(r["D"])]) if any(
                math.isfinite(r["D"]) for r in rows) else float("nan"))
            Ns.append(N)

        if len(Ns) < 2:
            continue
        Ns_arr = np.asarray(Ns, dtype=np.float64)
        log_N = np.log(Ns_arr)

        for tag, vals in (("T1", vals_R2), ("T2", vals_Rg2),
                          ("T6", vals_tau), ("T7", vals_D)):
            arr = np.asarray(vals, dtype=np.float64)
            mask = np.isfinite(arr) & (arr > 0)
            if mask.sum() < 2:
                out.append({"app": app, "phi": phi, "t_item": tag,
                            "fitted": float("nan"), "r2": float("nan"),
                            "in_range": False})
                continue
            slope, intercept = np.polyfit(log_N[mask], np.log(arr[mask]), 1)
            pred = slope * log_N[mask] + intercept
            ss_res = ((np.log(arr[mask]) - pred) ** 2).sum()
            ss_tot = ((np.log(arr[mask]) - np.log(arr[mask]).mean()) ** 2).sum()
            r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
            target = KURIATA_TARGETS[tag][0]
            in_range = (abs(slope - target) / abs(target) <= T_TOLERANCE
                         and r2 >= T_MIN_R2)
            out.append({"app": app, "phi": phi, "t_item": tag,
                        "fitted": float(slope), "r2": float(r2),
                        "in_range": bool(in_range)})

        # T3: ratio at N=100.
        if 100 in per_N:
            r2_at_100 = np.median([r["mean_R2"] for r in per_N[100]])
            rg2_at_100 = np.median([r["mean_Rg2"] for r in per_N[100]])
            ratio = r2_at_100 / max(rg2_at_100, 1e-30)
            target = KURIATA_TARGETS["T3"][0]
            in_range = abs(ratio - target) / abs(target) <= T_TOLERANCE
            out.append({"app": app, "phi": phi, "t_item": "T3",
                        "fitted": float(ratio), "r2": float("nan"),
                        "in_range": bool(in_range)})

        # T4: median of per-cell log-log gCM(t) slope across all (N, seed).
        # T5: median of per-cell short-time log-log g1(t) slope.
        for tag, key in (("T4", "gCM_slope"), ("T5", "g1_short_slope")):
            slopes = [r.get(key, float("nan"))
                       for N in per_N for r in per_N[N]]
            slopes = [s for s in slopes if math.isfinite(s)]
            if not slopes:
                out.append({"app": app, "phi": phi, "t_item": tag,
                            "fitted": float("nan"), "r2": float("nan"),
                            "in_range": False})
                continue
            med = float(np.median(slopes))
            target = KURIATA_TARGETS[tag][0]
            in_range = abs(med - target) / abs(target) <= T_TOLERANCE
            out.append({"app": app, "phi": phi, "t_item": tag,
                        "fitted": med, "r2": float("nan"),
                        "in_range": bool(in_range)})
    return out


def write_t_results(rows: List[Dict], verif_root: Path) -> None:
    path = verif_root / "t_results.tsv"
    fieldnames = ["app", "phi", "t_item", "fitted", "r2", "in_range"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_observables_summary(obs_rows: List[Dict], verif_root: Path) -> None:
    path = verif_root / "observables_summary.tsv"
    fieldnames = ["app", "N", "phi", "seed", "mean_R2", "std_R2",
                  "mean_Rg2", "std_Rg2", "tau_R", "D",
                  "gCM_slope", "g1_short_slope",
                  "wall_eq", "wall_prod"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in obs_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def compute_c7_wall_ratios(obs_rows: List[Dict]) -> List[Dict]:
    """C7 acceptance: wall_ratio(N=100/N=25) <= 20.0 per (app, phi)."""
    by_key: Dict[Tuple[str, float, int], Dict[int, float]] = {}
    for r in obs_rows:
        wall = r["wall_eq"] + r["wall_prod"]
        if not math.isfinite(wall):
            continue
        by_key.setdefault((r["app"], r["phi"], r["seed"]), {})[r["N"]] = wall

    out: List[Dict] = []
    for (app, phi, seed), per_N in by_key.items():
        if 25 in per_N and 100 in per_N and per_N[25] > 0:
            ratio = per_N[100] / per_N[25]
            out.append({"app": app, "phi": phi, "seed": seed,
                        "wall_ratio_N100_over_N25": ratio,
                        "passes_C7": ratio <= 20.0})
    return out


def render_plots(verif_root: Path, obs_rows: List[Dict], t_rows: List[Dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[verify_runner] matplotlib import failed: {e}")
        return

    plot_dir = verif_root / "plots"
    (plot_dir / "01_static").mkdir(parents=True, exist_ok=True)
    (plot_dir / "02_dynamic").mkdir(parents=True, exist_ok=True)
    (plot_dir / "03_equilibration").mkdir(parents=True, exist_ok=True)

    by_app_phi: Dict[Tuple[str, float], Dict[int, Dict[str, List[float]]]] = {}
    for r in obs_rows:
        d = by_app_phi.setdefault((r["app"], r["phi"]), {}).setdefault(r["N"], {})
        for k in ("mean_R2", "mean_Rg2", "tau_R", "D"):
            d.setdefault(k, []).append(r[k])

    def _agg(d: Dict[int, Dict[str, List[float]]], key: str):
        Ns_sorted = sorted(d.keys())
        means = [np.median([v for v in d[n].get(key, []) if math.isfinite(v)] or [float("nan")])
                 for n in Ns_sorted]
        stds = [np.std([v for v in d[n].get(key, []) if math.isfinite(v)] or [0.0])
                for n in Ns_sorted]
        return Ns_sorted, means, stds

    # T1 — <R^2> vs N.
    fig, ax = plt.subplots(figsize=(8, 6))
    for (app, phi), d in by_app_phi.items():
        Ns, means, stds = _agg(d, "mean_R2")
        ax.errorbar(Ns, means, yerr=stds, label=f"{app[:30]} phi={phi}",
                    marker='o', markersize=3, capsize=3, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N"); ax.set_ylabel("<R²>")
    ax.set_title("T1: <R²> vs N (Kuriata target slope 1.18)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_static" / "fig_R2_vs_N_per_phi.png", dpi=100)
    plt.close(fig)

    # T2 — <Rg^2> vs N.
    fig, ax = plt.subplots(figsize=(8, 6))
    for (app, phi), d in by_app_phi.items():
        Ns, means, stds = _agg(d, "mean_Rg2")
        ax.errorbar(Ns, means, yerr=stds, label=f"{app[:30]} phi={phi}",
                    marker='o', markersize=3, capsize=3, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N"); ax.set_ylabel("<Rg²>")
    ax.set_title("T2: <Rg²> vs N (Kuriata target slope 1.18)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_static" / "fig_Rg2_vs_N_per_phi.png", dpi=100)
    plt.close(fig)

    # T3 — ratio R2/Rg2 vs phi at N=100.
    fig, ax = plt.subplots(figsize=(8, 6))
    apps = sorted(set(a for a, _ in by_app_phi))
    for app in apps:
        phis = []; ratios = []
        for phi in PHI_VALUES:
            if (app, phi) in by_app_phi and 100 in by_app_phi[(app, phi)]:
                d100 = by_app_phi[(app, phi)][100]
                r2 = np.median([v for v in d100.get("mean_R2", []) if math.isfinite(v)] or [float("nan")])
                rg2 = np.median([v for v in d100.get("mean_Rg2", []) if math.isfinite(v)] or [float("nan")])
                phis.append(phi); ratios.append(r2 / max(rg2, 1e-30))
        if phis:
            ax.plot(phis, ratios, marker='o', label=app[:30], alpha=0.6)
    ax.axhline(6.25, color='k', ls='--', alpha=0.5, label="Kuriata 6.25")
    ax.set_xlabel("phi"); ax.set_ylabel("<R²>/<Rg²> at N=100")
    ax.set_title("T3: ratio at N=100")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_static" / "fig_ratio_R2_Rg2_vs_phi.png", dpi=100)
    plt.close(fig)

    # T6 — tau_R vs N.
    fig, ax = plt.subplots(figsize=(8, 6))
    for (app, phi), d in by_app_phi.items():
        Ns, means, stds = _agg(d, "tau_R")
        ax.errorbar(Ns, means, yerr=stds, label=f"{app[:30]} phi={phi}",
                    marker='o', markersize=3, capsize=3, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N"); ax.set_ylabel("tau_R")
    ax.set_title("T6: tau_R vs N")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_dynamic" / "fig_tauR_vs_N_per_phi.png", dpi=100)
    plt.close(fig)

    # T7 — D vs N.
    fig, ax = plt.subplots(figsize=(8, 6))
    for (app, phi), d in by_app_phi.items():
        Ns, means, stds = _agg(d, "D")
        ax.errorbar(Ns, means, yerr=stds, label=f"{app[:30]} phi={phi}",
                    marker='o', markersize=3, capsize=3, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N"); ax.set_ylabel("D")
    ax.set_title("T7: D vs N")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_dynamic" / "fig_D_vs_N_per_phi.png", dpi=100)
    plt.close(fig)

    # T4 — gCM(t) curves: <|R_cm(t) - R_cm(0)|^2> from trajectory PDBs.
    # T5 — g1(t) curves: <|r_mid(t) - r_mid(0)|^2> from trajectory PDBs.
    # C31/equilibration — R²(sweep), Rg²(sweep) from eq_observables.tsv +
    # prod_observables.tsv (one curve per (app, N, phi, seed)).
    _render_dynamic_and_eq_plots(verif_root, obs_rows, plot_dir, plt)


def _render_dynamic_and_eq_plots(verif_root: Path, obs_rows: List[Dict],
                                 plot_dir: Path, plt) -> None:
    """Compute gCM(t), g1(t) per cell from trajectory; build T4/T5 overlays.
    Read eq/prod_observables.tsv for R²(sweep) and Rg²(sweep) overlays."""
    fig_gcm, ax_gcm = plt.subplots(figsize=(8, 6))
    fig_g1, ax_g1 = plt.subplots(figsize=(8, 6))
    fig_r2, ax_r2 = plt.subplots(figsize=(8, 6))
    fig_rg2, ax_rg2 = plt.subplots(figsize=(8, 6))

    n_dyn = 0; n_eq = 0
    for r in obs_rows:
        cell_path = (verif_root / r["app"]
                     / f"N{r['N']}_phi{r['phi']:.2f}_seed{r['seed']}")
        # Dynamic curves from trajectory.
        gcm, g1 = _compute_gcm_g1(cell_path, r["N"])
        if gcm is not None:
            t = np.arange(1, len(gcm) + 1, dtype=np.float64)
            label = f"{r['app'][:24]} N={r['N']} phi={r['phi']}"
            ax_gcm.loglog(t, gcm, alpha=0.4, label=label if n_dyn < 12 else None)
            if g1 is not None:
                ax_g1.loglog(t, g1, alpha=0.4, label=label if n_dyn < 12 else None)
            n_dyn += 1

        # Equilibration: R²(sweep), Rg²(sweep) from eq + prod observables.
        for tsv_name, phase in (("eq_observables.tsv", "eq"),
                                 ("prod_observables.tsv", "prod")):
            tsv = cell_path / tsv_name
            if not tsv.exists():
                continue
            sweeps, r2, rg2 = _read_observables_tsv(tsv)
            if not sweeps:
                continue
            offset = 0 if phase == "eq" else max(sweeps) + 1 - len(sweeps) - 1
            xs = [s + (0 if phase == "eq" else 0) for s in sweeps]  # raw sweep idx
            ax_r2.plot(xs, r2, alpha=0.3,
                       label=f"{r['app'][:18]} N={r['N']} phi={r['phi']} {phase}"
                       if n_eq < 8 else None)
            ax_rg2.plot(xs, rg2, alpha=0.3,
                        label=f"{r['app'][:18]} N={r['N']} phi={r['phi']} {phase}"
                        if n_eq < 8 else None)
            n_eq += 1

    ax_gcm.set_xlabel("t (frame index, traj_stride sweeps)")
    ax_gcm.set_ylabel("gCM(t)")
    ax_gcm.set_title(f"T4: gCM(t) per cell (Kuriata slope 1.0); n={n_dyn} cells")
    ax_gcm.legend(fontsize=6, loc="best")
    fig_gcm.tight_layout()
    fig_gcm.savefig(plot_dir / "02_dynamic" / "fig_gCM_vs_sweep_per_phi.png", dpi=100)
    plt.close(fig_gcm)

    ax_g1.set_xlabel("t (frame index, traj_stride sweeps)")
    ax_g1.set_ylabel("g1(t) middle bead")
    ax_g1.set_title(f"T5: g1(t) per cell (Kuriata short-time slope 0.5); n={n_dyn} cells")
    ax_g1.legend(fontsize=6, loc="best")
    fig_g1.tight_layout()
    fig_g1.savefig(plot_dir / "02_dynamic" / "fig_g1_vs_sweep_per_phi.png", dpi=100)
    plt.close(fig_g1)

    ax_r2.set_xlabel("sweep")
    ax_r2.set_ylabel("R²")
    ax_r2.set_title(f"C31: R²(sweep) — eq + prod overlays; n={n_eq} curves")
    ax_r2.legend(fontsize=6, loc="best")
    fig_r2.tight_layout()
    fig_r2.savefig(plot_dir / "03_equilibration" / "fig_R2_vs_sweep_all_N.png", dpi=100)
    plt.close(fig_r2)

    ax_rg2.set_xlabel("sweep")
    ax_rg2.set_ylabel("Rg²")
    ax_rg2.set_title(f"C31: Rg²(sweep) — eq + prod overlays; n={n_eq} curves")
    ax_rg2.legend(fontsize=6, loc="best")
    fig_rg2.tight_layout()
    fig_rg2.savefig(plot_dir / "03_equilibration" / "fig_Rg2_vs_sweep_all_N.png", dpi=100)
    plt.close(fig_rg2)


def _compute_gcm_g1(cell_path: Path, N: int):
    """Returns (gCM_array, g1_array) or (None, None) if trajectory unusable."""
    traj = cell_path / "trajectory"
    if not traj.is_dir():
        return None, None
    snaps = sorted(traj.glob("snap_*.pdb"))
    if len(snaps) < 5:
        return None, None
    chains_per_frame: List[np.ndarray] = []
    for p in snaps:
        ca = read_pdb_ca(p, N)
        if ca is None:
            return None, None
        chains_per_frame.append(ca)
    arr = np.stack(chains_per_frame, axis=0)  # [T, n_chains, N, 3]
    R_cm = arr.mean(axis=2)                     # [T, n_chains, 3]
    mid = N // 2
    r_mid = arr[:, :, mid, :]                    # [T, n_chains, 3]
    gCM = np.array([(((R_cm[t] - R_cm[0]) ** 2).sum(axis=-1)).mean()
                    for t in range(arr.shape[0])])
    g1 = np.array([(((r_mid[t] - r_mid[0]) ** 2).sum(axis=-1)).mean()
                   for t in range(arr.shape[0])])
    # Drop t=0 (zero by construction; loglog can't render).
    return gCM[1:], g1[1:]


def _read_observables_tsv(path: Path):
    """Read sweep, R², Rg² columns. Returns (sweeps, r2_list, rg2_list)."""
    sweeps: List[int] = []
    r2: List[float] = []
    rg2: List[float] = []
    try:
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    sweeps.append(int(row.get("sweep", row.get("step", 0))))
                    r2.append(float(row.get("re2_mean", row.get("R2", float("nan")))))
                    rg2.append(float(row.get("rg2_mean", row.get("Rg2", float("nan")))))
                except (ValueError, KeyError):
                    continue
    except Exception:
        return [], [], []
    return sweeps, r2, rg2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaled", action="store_true",
                    help="Use scaled-down params (n_chains=10, eq=200, prod=500). "
                         "Default is the spec from context_rouse_verification.txt §8.")
    ap.add_argument("--output_dir", default="_verif_full",
                    help="Verification root directory")
    ap.add_argument("--apps", default=None,
                    help="Comma-separated app subset; default = all 16")
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent CPU subprocesses (GPU is serialized)")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    apps = args.apps.split(",") if args.apps else APPS_ALL
    apps = [a.strip() for a in apps if a.strip()]
    for a in apps:
        if a not in APPS_ALL:
            print(f"[verify_runner] unknown app: {a}", file=sys.stderr)
            return 2

    verif_root = (Path(__file__).resolve().parent.parent / args.output_dir).resolve()
    verif_root.mkdir(parents=True, exist_ok=True)
    mode = "SCALED" if args.scaled else "SPEC (n_chains=100, per-N schedule §8)"
    print(f"[verify_runner] verif_root = {verif_root}")
    print(f"[verify_runner] mode = {mode}")

    cells: List[Tuple] = []
    for app in apps:
        for N in N_VALUES:
            for phi in PHI_VALUES:
                for seed in SEEDS:
                    cells.append((app, N, phi, seed, verif_root, args.python, args.scaled))
    print(f"[verify_runner] {len(cells)} cells queued "
          f"({len(apps)} apps × {len(N_VALUES)} N × {len(PHI_VALUES)} phi × {len(SEEDS)} seeds)")

    cpu_cells = [c for c in cells if not is_gpu(c[0])]
    gpu_cells = [c for c in cells if is_gpu(c[0])]

    results: List[Dict] = []
    t0 = time.time()

    # CPU cells — process pool.
    if cpu_cells:
        print(f"[verify_runner] starting {len(cpu_cells)} CPU cells via "
              f"ProcessPoolExecutor(workers={args.workers})")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_cell, c): c for c in cpu_cells}
            for n, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                if n % 10 == 0 or r["rc"] != 0:
                    print(f"  [{n}/{len(cpu_cells)}] {r['app']} N={r['N']} "
                          f"phi={r['phi']} seed={r['seed']} rc={r['rc']} "
                          f"wall={r['wall_s']:.1f}s")

    # GPU cells — serial.
    if gpu_cells:
        print(f"[verify_runner] starting {len(gpu_cells)} GPU cells (serial)")
        for n, c in enumerate(gpu_cells, 1):
            r = run_cell(c)
            results.append(r)
            if n % 5 == 0 or r["rc"] != 0:
                print(f"  [{n}/{len(gpu_cells)}] {r['app']} N={r['N']} "
                      f"phi={r['phi']} seed={r['seed']} rc={r['rc']} "
                      f"wall={r['wall_s']:.1f}s")

    print(f"[verify_runner] cells finished in {(time.time() - t0) / 60.0:.1f} min")

    n_ok = sum(1 for r in results if r["rc"] == 0)
    print(f"[verify_runner] {n_ok}/{len(results)} cells succeeded")

    # Aggregate.
    print("[verify_runner] aggregating c_results.tsv …")
    c_rows = aggregate_c_results(verif_root, results)
    write_c_results(c_rows, verif_root)

    print("[verify_runner] aggregating observables_summary.tsv …")
    obs_rows: List[Dict] = []
    for cell in results:
        if cell["rc"] != 0:
            continue
        try:
            obs_rows.append(cell_observables(verif_root, cell))
        except Exception as e:
            print(f"  obs read fail for {cell['app']} N={cell['N']}: {e}")
    write_observables_summary(obs_rows, verif_root)

    print("[verify_runner] computing T-fits …")
    t_rows = fit_t_items(obs_rows)
    write_t_results(t_rows, verif_root)

    c7_rows = compute_c7_wall_ratios(obs_rows)
    with (verif_root / "c7_wall_ratios.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["app", "phi", "seed",
                                          "wall_ratio_N100_over_N25", "passes_C7"],
                            delimiter="\t")
        w.writeheader(); w.writerows(c7_rows)

    print("[verify_runner] rendering plots …")
    render_plots(verif_root, obs_rows, t_rows)

    print(f"[verify_runner] done. Outputs in {verif_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
