"""Bench: cell-list apps + reference CUDA-C at large K.

Apps:
  - g1c_ptgcl, gnc_ptgcl, g1m_ptgcl, gnm_ptgcl  (cell-list, all 4 corners)
  - g1c_ptg                                       (all-pairs reference at K=1000;
                                                    will likely time out)
  - g1c_cc                                        (CUDA-C all-pairs reference)

Cells:
  K=1000 (N=100)  — where cell-list payoff kicks in.

Per-cell timeout: 600s.
"""
from __future__ import annotations
import csv, json, math, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "_bench_large_K_cell_list"
OUT.mkdir(parents=True, exist_ok=True)

CELL_LIST_APPS = [
    "g1c_ptgcl_gpu_single_thread_conventional_mc_py_torch_gpu_fused_cell_list",
    "gnc_ptgcl_gpu_multi_thread_conventional_mc_py_torch_gpu_fused_cell_list",
    "g1m_ptgcl_gpu_single_thread_multistep_mc_py_torch_gpu_fused_cell_list",
    "gnm_ptgcl_gpu_multi_thread_multistep_mc_py_torch_gpu_fused_cell_list",
]
REF_APPS = [
    "g1c_ptg_gpu_single_thread_conventional_mc_py_torch_gpu_fused",
    "g1c_cc_gpu_single_thread_conventional_mc_cuda_c",
]
APPS = CELL_LIST_APPS + REF_APPS

N = 100
K = 1000
EQ = 2
PROD = 3
PHI = 0.10
SEED = 42
BATCH_SIZE_MULTISTEP = 256
N_STREAMS = 4
TIMEOUT_S = 600


def is_pt_fused(app): return "py_torch_gpu_fused" in app
def is_multistep(app): return "multistep" in app
def is_multi_thread(app): return "multi_thread" in app


def build_cmd(app, cell_out):
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
    if is_multi_thread(app):
        cmd += ["--streams", str(N_STREAMS)]
    return cmd


def run_one(app):
    cell_out = OUT / app
    cell_out.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(app, cell_out)
    env = os.environ.copy()
    if is_pt_fused(app):
        env["ROUSE_TORCH_COMPILE_MODE"] = "inductor"
    t0 = time.perf_counter()
    status = "OK"
    rc = -999
    err_tail = ""
    try:
        cp = subprocess.run(cmd, cwd=str(SRC), capture_output=True, text=True,
                             timeout=TIMEOUT_S, env=env)
        rc = cp.returncode
        wall = time.perf_counter() - t0
        if rc != 0:
            status = "FAIL"
            err_tail = (cp.stderr or "")[-400:].replace("\n", " | ")
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        rc = -9
        status = "TIMEOUT"
        err_tail = f"timeout {TIMEOUT_S}s"

    per_sweep_s = float("nan")
    re2_mean = float("nan")
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
        except Exception as e:
            err_tail = (err_tail + f" | parse: {e}")[-400:]
    return {
        "app": app, "rc": rc, "wall_s_subprocess": round(wall, 2),
        "per_sweep_s": per_sweep_s, "re2_mean": float(re2_mean) if isinstance(re2_mean, (int, float)) else float("nan"),
        "status": status, "err_tail": err_tail,
    }


def main():
    rows = []
    t_start = time.perf_counter()
    for i, app in enumerate(APPS, 1):
        print(f"  [{i:02d}/{len(APPS)}] app={app} ...", flush=True)
        r = run_one(app)
        rows.append(r)
        per = f"{r['per_sweep_s']:.4g}s" if math.isfinite(r['per_sweep_s']) else "n/a"
        print(f"    -> status={r['status']} wall={r['wall_s_subprocess']:.1f}s per_sweep={per}", flush=True)
    print(f"\n[done] {len(APPS)} apps in {time.perf_counter()-t_start:.1f}s")

    with (OUT / "results.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["app", "per_sweep_s", "re2_mean", "wall_s_subprocess", "status", "err_tail"], delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    print("\n" + "=" * 88)
    print(f" HEADLINE — N={N} K={K} eq={EQ} prod={PROD} (sorted by per_sweep_s ascending) ")
    print("=" * 88)
    finite = sorted((r for r in rows if math.isfinite(r['per_sweep_s'])),
                    key=lambda r: r['per_sweep_s'])
    failed = [r for r in rows if not math.isfinite(r['per_sweep_s'])]
    for rank, r in enumerate(finite, 1):
        print(f"  {rank:>3}  {r['app']:<70} {r['per_sweep_s']:>10.4g}s  re2={r['re2_mean']:.1f}")
    if failed:
        print("  --- failed/timeout ---")
        for r in failed:
            print(f"        {r['app']:<70} status={r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
