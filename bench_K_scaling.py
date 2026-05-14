"""K-scaling crossover test: does PyTorch CPU finally beat Numba CPU when
n_chains is pushed to thousands?

Pair: c1c_nb_cpu_single_core_conventional_mc_numba vs c1c_pt_cpu_single_core_conventional_mc_py_torch
Fixed: N=25, phi=0.10, eq=10, prod=10, seed=42.
Swept: K ∈ {10, 50, 100, 250, 500, 1000}.

Output: _bench_K_scaling/bench_K_scaling.tsv with columns
  K, t_numba, t_pytorch, ratio_pt_over_nb
"""
from __future__ import annotations
import csv, json, math, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "_bench_K_scaling"
OUT.mkdir(parents=True, exist_ok=True)

APPS = [
    "c1c_nb_cpu_single_core_conventional_mc_numba",
    "c1c_pt_cpu_single_core_conventional_mc_py_torch",
]
K_VALUES = [10, 50, 100, 250, 500, 1000]
N = 25
PHI = 0.10
EQ = 10
PROD = 10
SEED = 42


def run_cell(app: str, K: int):
    cell_out = OUT / app / f"K{K}"
    cell_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", f"{app}.main",
        "--N", str(N), "--n_chains", str(K), "--phi", f"{PHI:.6f}",
        "--eq_sweeps", str(EQ), "--prod_sweeps", str(PROD),
        "--seed", str(SEED), "--init_method", "random_saw",
        "--residues_per_segment", "8", "--max_angle_hinge_pi", "0.5",
        "--traj_stride", "0",
        "--contact_energy", "0.0", "--repulsive_energy", "1e6",
        "--output_dir", str(cell_out),
        "--log_level", "INFO",
        "--log_file", str(cell_out / "run.log"),
    ]
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(cmd, cwd=str(SRC), capture_output=True, text=True,
                             timeout=3600)
        wall = time.perf_counter() - t0
        rc = cp.returncode
        err_tail = (cp.stderr or "").strip().splitlines()[-3:]
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        rc = -9
        err_tail = ["timeout 3600s"]
    per_sweep = float("nan")
    sj = cell_out / "summary.json"
    if sj.exists():
        try:
            d = json.loads(sj.read_text(encoding="utf-8"))
            timing = d.get("timing", {}) or {}
            for key in ("mean_per_sweep_s", "per_sweep_s_prod", "per_sweep_s",
                        "prod_per_sweep_s"):
                if key in timing:
                    per_sweep = float(timing[key])
                    break
            if not math.isfinite(per_sweep):
                tot = timing.get("total_wall_s") or timing.get("prod_wall_s")
                if tot is not None and math.isfinite(float(tot)):
                    per_sweep = float(tot) / (EQ + PROD)
        except Exception as e:
            err_tail.append(f"parse: {e}")
    return {"app": app, "K": K, "rc": rc, "wall_s": round(wall, 2),
            "per_sweep_s": per_sweep, "stderr_tail": " | ".join(err_tail)}


def main():
    print(f"[K-scale] N={N} K_values={K_VALUES} eq={EQ} prod={PROD}")
    rows = []
    for K in K_VALUES:
        for app in APPS:
            print(f"  {app} K={K} ...", flush=True)
            r = run_cell(app, K)
            print(f"    rc={r['rc']} wall={r['wall_s']:.1f}s "
                  f"per_sweep={r['per_sweep_s']:.4g}s "
                  f"err={r['stderr_tail'][:120]}")
            rows.append(r)
    # Build comparison
    by_K = {}
    for r in rows:
        by_K.setdefault(r["K"], {})[r["app"]] = r["per_sweep_s"]
    summary_path = OUT / "bench_K_scaling.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["K", "t_numba_s", "t_pytorch_s", "ratio_pt_over_nb"])
        for K in K_VALUES:
            tn = by_K.get(K, {}).get("c1c_nb_cpu_single_core_conventional_mc_numba",
                                     float("nan"))
            tp = by_K.get(K, {}).get("c1c_pt_cpu_single_core_conventional_mc_py_torch",
                                     float("nan"))
            ratio = tp / tn if math.isfinite(tn) and tn > 0 else float("nan")
            w.writerow([K, tn, tp, ratio])
    print(f"[K-scale] wrote {summary_path}")
    print("K\tt_numba\tt_pytorch\tratio_pt/nb")
    for K in K_VALUES:
        tn = by_K.get(K, {}).get("c1c_nb_cpu_single_core_conventional_mc_numba",
                                 float("nan"))
        tp = by_K.get(K, {}).get("c1c_pt_cpu_single_core_conventional_mc_py_torch",
                                 float("nan"))
        ratio = tp / tn if math.isfinite(tn) and tn > 0 else float("nan")
        print(f"{K}\t{tn:.4g}\t{tp:.4g}\t{ratio:.2f}x")


if __name__ == "__main__":
    main()
