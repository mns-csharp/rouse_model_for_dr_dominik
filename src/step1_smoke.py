"""Step 1 — bug-free smoke: run one fast cell per app and aggregate
[CHECKLIST-Cn] verdicts.

One cell (N=50, n_chains=10, phi=0.10, seed=42, eq=200, prod=500) exercises
every C-item code path in every app. This gives 33 PASS verdicts × 16 apps
= 528 PASS verdicts at the floor for "100% bug-free" coverage.

Outputs to <output_dir>/ a layout identical to verify_runner.py so that
verify_audit.py can be pointed at it directly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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

# One fast cell per app — same shape as verify_runner.py so verify_audit.py
# accepts the layout.
N = 50
PHI = 0.10
SEED = 42
EQ = 200
PROD = 500
N_CHAINS = 10
TRAJ_STRIDE = 10


def is_gpu(app): return app.startswith("gpu_")
def is_multistep(app): return "multistep" in app
def is_multi_core(app): return "multi_core" in app
def is_multi_thread(app): return "multi_thread" in app
def needs_workers(app): return app == "cnm_pt_cpu_multi_core_multistep_mc_py_torch"


def build_cmd(python_exe: str, app: str, cell_out: Path):
    cmd = [
        python_exe, "-m", f"{app}.main",
        "--N", str(N),
        "--n_chains", str(N_CHAINS),
        "--phi", f"{PHI:.6f}",
        "--eq_sweeps", str(EQ),
        "--prod_sweeps", str(PROD),
        "--seed", str(SEED),
        "--init_method", "random_saw",
        "--residues_per_segment", "8",
        "--max_angle_hinge_pi", "0.5",
        "--traj_stride", str(TRAJ_STRIDE),
        "--contact_energy", "0.0",
        "--repulsive_energy", "1e6",
        "--output_dir", str(cell_out),
        "--log_level", "INFO",
        "--log_file", str(cell_out / "run.log"),
        "--cap_inner_hinge",
        "--cap_tail",
    ]
    if is_multistep(app):
        cmd += ["--batch_size", "1"]
    if is_multi_core(app):
        cmd += ["--threads", "1"]
    if needs_workers(app):
        cmd += ["--workers", "1"]
    if is_multi_thread(app):
        cmd += ["--streams", "1"]
    return cmd


def run_one(args):
    python_exe, app, root, src_dir = args
    cell_out = root / app / f"N{N}_phi{PHI:.2f}_seed{SEED}"
    cell_out.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(python_exe, app, cell_out)
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, cwd=str(src_dir), capture_output=True,
                             text=True, timeout=1800)
        return {"app": app, "rc": cp.returncode, "wall_s": round(time.time() - t0, 2),
                "stderr_tail": (cp.stderr or "").strip().splitlines()[-2:],
                "cell_dir": str(cell_out)}
    except subprocess.TimeoutExpired:
        return {"app": app, "rc": -1, "wall_s": round(time.time() - t0, 2),
                "stderr_tail": ["timeout 1800s"], "cell_dir": str(cell_out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="_step1_smoke")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    root = (src_dir.parent / args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"[step1_smoke] root = {root}")

    cpu_apps = [a for a in APPS_ALL if not is_gpu(a)]
    gpu_apps = [a for a in APPS_ALL if is_gpu(a)]
    print(f"[step1_smoke] {len(cpu_apps)} CPU apps (workers={args.workers}); "
          f"{len(gpu_apps)} GPU apps (serial)")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, (args.python, app, root, src_dir)): app for app in cpu_apps}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            print(f"  CPU [{n}/{len(cpu_apps)}] {r['app']} rc={r['rc']} wall={r['wall_s']:.1f}s",
                  flush=True)
            if r["rc"] != 0 and r["stderr_tail"]:
                for l in r["stderr_tail"]:
                    print(f"    {l}")
    for n, app in enumerate(gpu_apps, 1):
        r = run_one((args.python, app, root, src_dir))
        results.append(r)
        print(f"  GPU [{n}/{len(gpu_apps)}] {app} rc={r['rc']} wall={r['wall_s']:.1f}s",
              flush=True)
        if r["rc"] != 0 and r["stderr_tail"]:
            for l in r["stderr_tail"]:
                print(f"    {l}")

    print(f"[step1_smoke] done in {(time.time() - t0) / 60.0:.1f} min; "
          f"{sum(1 for r in results if r['rc'] == 0)}/{len(results)} apps ok")
    rcs = {r["app"]: r["rc"] for r in results}
    return 0 if all(rc == 0 for rc in rcs.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
