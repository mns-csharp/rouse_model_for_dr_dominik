"""Sweep N at K=10 to test the claim that PyTorch CPU will close the gap on
Numba CPU as the per-op tensor work grows.

Pair: c1c_nb_cpu_single_core_conventional_mc_numba  vs c1c_pt_cpu_single_core_conventional_mc_py_torch.
N: 25, 50, 100, 150, 200, 300, 400.
eq=20, prod=20, n_chains=10.

Output: bench_pt_vs_nb.tsv with columns
  N, t_numba, t_pytorch, ratio_pt_over_nb
"""
from __future__ import annotations
import csv, json, math, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "_bench_pt_vs_nb"
OUT.mkdir(parents=True, exist_ok=True)

APPS = [
    "c1c_nb_cpu_single_core_conventional_mc_numba",
    "c1c_pt_cpu_single_core_conventional_mc_py_torch",
]
N_VALUES = [25, 50, 100, 150, 200, 300, 400]
EQ = 20
PROD = 20
K = 10
PHI = 0.10


def run_cell(app, N):
    cell_out = OUT / app / f"N{N}"
    cell_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", f"{app}.main",
        "--N", str(N), "--n_chains", str(K), "--phi", f"{PHI:.6f}",
        "--eq_sweeps", str(EQ), "--prod_sweeps", str(PROD),
        "--seed", "42", "--init_method", "random_saw",
        "--residues_per_segment", "8", "--max_angle_hinge_pi", "0.5",
        "--traj_stride", "0",
        "--contact_energy", "0.0", "--repulsive_energy", "1e6",
        "--output_dir", str(cell_out),
        "--log_level", "INFO",
        "--log_file", str(cell_out / "run.log"),
    ]
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, cwd=str(SRC), capture_output=True, text=True,
                         timeout=1800)
    wall = time.perf_counter() - t0
    rc = cp.returncode
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
                if tot and math.isfinite(float(tot)):
                    per_sweep = float(tot) / (EQ + PROD)
        except Exception:
            pass
    return {"app": app, "N": N, "rc": rc, "wall_s": round(wall, 2),
            "per_sweep_s": per_sweep}


def main():
    print(f"[pt-vs-nb] N values: {N_VALUES}, K={K}, eq={EQ}, prod={PROD}")
    rows = []
    for N in N_VALUES:
        for app in APPS:
            print(f"  {app} N={N} ...", flush=True)
            r = run_cell(app, N)
            print(f"    rc={r['rc']} wall={r['wall_s']:.1f}s "
                  f"per_sweep={r['per_sweep_s']:.4g}s")
            rows.append(r)

    # Build comparison table
    by_N = {}
    for r in rows:
        by_N.setdefault(r["N"], {})[r["app"]] = r["per_sweep_s"]

    summary_path = OUT / "bench_pt_vs_nb.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["N", "t_numba_s", "t_pytorch_s", "ratio_pt_over_nb"])
        for N in N_VALUES:
            tn = by_N.get(N, {}).get("c1c_nb_cpu_single_core_conventional_mc_numba",
                                     float("nan"))
            tp = by_N.get(N, {}).get("c1c_pt_cpu_single_core_conventional_mc_py_torch",
                                     float("nan"))
            ratio = tp / tn if math.isfinite(tn) and tn > 0 else float("nan")
            w.writerow([N, tn, tp, ratio])
    print(f"[pt-vs-nb] wrote {summary_path}")
    # Echo table
    print("N\tt_numba\tt_pytorch\tratio_pt/nb")
    for N in N_VALUES:
        tn = by_N.get(N, {}).get("c1c_nb_cpu_single_core_conventional_mc_numba",
                                 float("nan"))
        tp = by_N.get(N, {}).get("c1c_pt_cpu_single_core_conventional_mc_py_torch",
                                 float("nan"))
        ratio = tp / tn if math.isfinite(tn) and tn > 0 else float("nan")
        print(f"{N}\t{tn:.4g}\t{tp:.4g}\t{ratio:.2f}x")


if __name__ == "__main__":
    main()
