"""Bench: does GPU-PyTorch ever beat CPU-Numba?

Sweep 6 (N, K) cells × 8 apps (4 CPU-Numba + 4 GPU-PT).
- Conventional apps run at default batch (effective B=1).
- Multistep apps run with --batch_size 64.
- Per-cell timeout: 300 s.

Outputs:
  _bench_gpu_pt_vs_cpu_numba/<app>/<cell_id>/summary.json     (per-cell artefacts)
  _bench_gpu_pt_vs_cpu_numba/results.tsv                       (one row per (cell, app))
  _bench_gpu_pt_vs_cpu_numba/comparison.tsv                    (one row per cell)
"""
from __future__ import annotations
import csv, json, math, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "_bench_gpu_pt_vs_cpu_numba"
OUT.mkdir(parents=True, exist_ok=True)

CPU_NUMBA_APPS = [
    "c1c_nb_cpu_single_core_conventional_mc_numba",
    "c1m_nb_cpu_single_core_multistep_mc_numba",
    "cnc_nb_cpu_multi_core_conventional_mc_numba",
    "cnm_nb_cpu_multi_core_multistep_mc_numba",
]
GPU_PT_APPS = [
    "g1c_pt_gpu_single_thread_conventional_mc_py_torch",
    "g1m_pt_gpu_single_thread_multistep_mc_py_torch",
    "gnc_pt_gpu_multi_thread_conventional_mc_py_torch",
    "gnm_pt_gpu_multi_thread_multistep_mc_py_torch",
]
APPS = CPU_NUMBA_APPS + GPU_PT_APPS

CELLS = [
    # Two cells. Subprocess wall is dominated by per-process init (SAW chain
    # placement + numba JIT compile = ~30-60 s at K=100, ~150 s at K=500),
    # not by per-sweep cost. So even the "small" cell has substantial fixed
    # overhead; we keep cell count small and timeout generous.
    ("C1", 50,  100),    # NK=5,000  — tiny, baseline for both backends.
    ("C2", 50,  500),    # NK=25,000 — large K at small N, best chance for GPU-PT to amortise.
]

# 3-sweep runs (eq=1 prod=2) are enough to get a stable per_sweep_s after
# numba's ~3 s JIT warmup. Wall budget per cell is then ~K-bound.
EQ = 1
PROD = 2
PHI = 0.10
SEED = 42
BATCH_SIZE_MULTISTEP = 64
# 600 s per cell so even K=500 numba (init ~150 s + JIT ~40 s + sweeps ~30 s)
# completes. GPU-PT-multistep is expected to TIMEOUT — iter2 confirmed it
# blows past 1800 s at much smaller cells; recording TIMEOUT is the answer.
TIMEOUT_S = 600


def is_multistep(app: str) -> bool: return "multistep" in app
def is_multi_core(app: str) -> bool: return "multi_core" in app
def is_multi_thread(app: str) -> bool: return "multi_thread" in app
def needs_workers(app: str) -> bool: return app == "cnm_pt_cpu_multi_core_multistep_mc_py_torch"


def build_cmd(app: str, N: int, K: int, cell_out: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", f"{app}.main",
        "--N", str(N), "--n_chains", str(K), "--phi", f"{PHI:.6f}",
        "--eq_sweeps", str(EQ), "--prod_sweeps", str(PROD),
        "--seed", str(SEED), "--init_method", "random_saw",
        "--residues_per_segment", "8", "--max_angle_hinge_pi", "0.5",
        "--traj_stride", "0",
        "--contact_energy", "0.0", "--repulsive_energy", "1e6",
        "--output_dir", str(cell_out),
        "--log_level", "WARNING",
        "--log_file", str(cell_out / "run.log"),
    ]
    if is_multistep(app):
        cmd += ["--batch_size", str(BATCH_SIZE_MULTISTEP)]
    if is_multi_core(app):
        cmd += ["--threads", "1"]
    if needs_workers(app):
        cmd += ["--workers", "1"]
    if is_multi_thread(app):
        cmd += ["--streams", "1"]
    return cmd


def run_one(app: str, cell_id: str, N: int, K: int) -> dict:
    cell_out = OUT / app / cell_id
    cell_out.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(app, N, K, cell_out)
    t0 = time.perf_counter()
    status = "OK"
    rc = -999
    err_tail = ""
    try:
        cp = subprocess.run(cmd, cwd=str(SRC), capture_output=True, text=True,
                             timeout=TIMEOUT_S)
        rc = cp.returncode
        wall = time.perf_counter() - t0
        if rc != 0:
            status = "FAIL"
            err_tail = (cp.stderr or "")[-300:].replace("\n", " | ")
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        rc = -9
        status = "TIMEOUT"
        err_tail = f"timeout {TIMEOUT_S}s"

    per_sweep_s = float("nan")
    re2_mean = float("nan")
    rg2_mean = float("nan")
    sj = cell_out / "summary.json"
    if sj.exists():
        try:
            d = json.loads(sj.read_text(encoding="utf-8"))
            timing = d.get("timing", {}) or {}
            obs = d.get("observables", {}) or {}
            prod_wall = timing.get("prod_wall_s")
            if prod_wall and math.isfinite(float(prod_wall)) and float(prod_wall) > 0:
                per_sweep_s = float(prod_wall) / PROD
            re2_mean = obs.get("re2_mean", float("nan"))
            rg2_mean = obs.get("rg2_mean", float("nan"))
        except Exception as e:
            err_tail = (err_tail + f" | parse: {e}")[-300:]
    return {
        "cell_id": cell_id, "N": N, "K": K, "app": app,
        "rc": rc, "wall_s_subprocess": round(wall, 2),
        "per_sweep_s": per_sweep_s,
        "re2_mean": re2_mean if isinstance(re2_mean, float) else float(re2_mean),
        "rg2_mean": rg2_mean if isinstance(rg2_mean, float) else float(rg2_mean),
        "status": status, "err_tail": err_tail,
    }


def write_tsv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    rows: list[dict] = []
    t_start = time.perf_counter()
    total_runs = len(CELLS) * len(APPS)
    n = 0
    for cell_id, N, K in CELLS:
        for app in APPS:
            n += 1
            tag = f"[{n:02d}/{total_runs}] cell={cell_id} N={N} K={K} app={app}"
            print(f"  {tag} ...", flush=True)
            r = run_one(app, cell_id, N, K)
            rows.append(r)
            print(f"    -> status={r['status']} rc={r['rc']} wall={r['wall_s_subprocess']:.1f}s "
                  f"per_sweep={r['per_sweep_s']:.4g}s re2={r['re2_mean']:.1f}")
    total_wall = time.perf_counter() - t_start
    print(f"\n[done] {total_runs} cells in {total_wall:.1f}s")

    results_path = OUT / "results.tsv"
    write_tsv(rows, results_path, [
        "cell_id", "N", "K", "app", "rc", "wall_s_subprocess",
        "per_sweep_s", "re2_mean", "rg2_mean", "status", "err_tail",
    ])
    print(f"[wrote] {results_path}")

    # Build the comparison TSV: one row per cell.
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(r["cell_id"], []).append(r)
    comp_rows = []
    for cell_id, N, K in CELLS:
        cell_rows = by_cell.get(cell_id, [])
        def best(group: list[str]) -> tuple[str, float]:
            best_app = ""
            best_t = float("inf")
            for r in cell_rows:
                if r["app"] not in group:
                    continue
                t = r["per_sweep_s"]
                if isinstance(t, float) and math.isfinite(t) and t > 0 and t < best_t:
                    best_t = t
                    best_app = r["app"]
            return best_app, best_t
        best_cpu_app, best_cpu_t = best(CPU_NUMBA_APPS)
        best_gpu_app, best_gpu_t = best(GPU_PT_APPS)
        ratio = (best_gpu_t / best_cpu_t) if (best_cpu_t > 0 and math.isfinite(best_cpu_t)
                                              and math.isfinite(best_gpu_t)) else float("nan")
        gpu_wins = (math.isfinite(ratio) and ratio < 1.0)
        comp_rows.append({
            "cell_id": cell_id, "N": N, "K": K,
            "best_cpu_numba_app": best_cpu_app or "n/a",
            "best_cpu_numba_t": best_cpu_t if math.isfinite(best_cpu_t) else float("nan"),
            "best_gpu_pt_app": best_gpu_app or "n/a",
            "best_gpu_pt_t": best_gpu_t if math.isfinite(best_gpu_t) else float("nan"),
            "ratio_gpu_pt_over_cpu_numba": ratio,
            "gpu_pt_wins": bool(gpu_wins),
        })
    comp_path = OUT / "comparison.tsv"
    write_tsv(comp_rows, comp_path, [
        "cell_id", "N", "K",
        "best_cpu_numba_app", "best_cpu_numba_t",
        "best_gpu_pt_app", "best_gpu_pt_t",
        "ratio_gpu_pt_over_cpu_numba", "gpu_pt_wins",
    ])
    print(f"[wrote] {comp_path}")
    print()
    print("=" * 80)
    print(" HEADLINE COMPARISON ")
    print("=" * 80)
    for r in comp_rows:
        win = "YES" if r["gpu_pt_wins"] else " no"
        ratio_str = (f"{r['ratio_gpu_pt_over_cpu_numba']:.2f}x"
                     if isinstance(r['ratio_gpu_pt_over_cpu_numba'], float)
                     and math.isfinite(r['ratio_gpu_pt_over_cpu_numba']) else "n/a")
        print(f"  {r['cell_id']} N={r['N']:>3} K={r['K']:>4}: "
              f"cpu_numba_best={r['best_cpu_numba_t']:.4g}s "
              f"gpu_pt_best={r['best_gpu_pt_t']:.4g}s "
              f"ratio={ratio_str}  GPU-PT wins? {win}")
    n_wins = sum(1 for r in comp_rows if r["gpu_pt_wins"])
    print()
    print(f"  GPU-PT wins {n_wins}/{len(comp_rows)} cells.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
