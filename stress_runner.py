"""Stress-test runner for the 16 Rouse apps.

Spec from /loop input:
  eq=100, prod=100, n_chains=10, N in {25, 50, 100}, phi=0.10, seed=42.

For each (app, N): launch `python -m <app>.main` once with the standard
flag set; capture wall_s (total) and per_sweep_s (from summary.json), plus
return code and stderr tail. Write stress_timings.tsv + bug-report blocks.

Bugs flagged automatically:
  (1) GPU CUDA-C app slower or equal vs analogous Numba/PyTorch app.
  (2) Multistep MC slower or equal vs analogous conventional MC app.
  (3) Crash / non-zero exit / NaN / Inf in summary.json (stress hits).

Usage:
    python stress_runner.py --output_dir _stress_iterN
"""
from __future__ import annotations
import argparse, csv, json, math, os, statistics, subprocess, sys, time
from pathlib import Path

APPS = [
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
    "g1c_ptg_gpu_single_thread_conventional_mc_py_torch_gpu_fused",
    "gnc_ptg_gpu_multi_thread_conventional_mc_py_torch_gpu_fused",
    "g1m_ptg_gpu_single_thread_multistep_mc_py_torch_gpu_fused",
    "gnm_ptg_gpu_multi_thread_multistep_mc_py_torch_gpu_fused",
    "g1c_ptgcl_gpu_single_thread_conventional_mc_py_torch_gpu_fused_cell_list",
    "g1m_ptgcl_gpu_single_thread_multistep_mc_py_torch_gpu_fused_cell_list",
    "gnc_ptgcl_gpu_multi_thread_conventional_mc_py_torch_gpu_fused_cell_list",
    "gnm_ptgcl_gpu_multi_thread_multistep_mc_py_torch_gpu_fused_cell_list",
    "g1c_cccl_gpu_single_thread_conventional_mc_cuda_c_cell_list",
]

N_VALUES = [25, 50, 100]
PHI = 0.10
N_CHAINS = 10
EQ = 100
PROD = 100
SEED = 42
TIMEOUT_S = 1800  # 30-min cap per cell


def is_multistep(app): return "multistep" in app
def is_multi_core(app): return "multi_core" in app
def is_multi_thread(app): return "multi_thread" in app
def is_gpu(app): return app.startswith("gpu_")
def is_cuda_c(app): return "cuda_c" in app
def is_pytorch(app): return "py_torch" in app
def is_numba(app): return "numba" in app


def build_cmd(python_exe, app, N, cell_out: Path, batch_size: int = 1):
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
        "--traj_stride", "0",
        "--contact_energy", "0.0",
        "--repulsive_energy", "1e6",
        "--output_dir", str(cell_out),
        "--log_level", "INFO",
        "--log_file", str(cell_out / "run.log"),
    ]
    if is_multistep(app):
        cmd += ["--batch_size", str(batch_size)]
    if is_multi_core(app):
        cmd += ["--threads", "1"]
    if app == "cnm_pt_cpu_multi_core_multistep_mc_py_torch":
        cmd += ["--workers", "1"]
    if is_multi_thread(app):
        cmd += ["--streams", "1"]
    return cmd


def run_one(app, N, root: Path, python_exe: str, batch_size: int = 1):
    cell_out = root / app / f"N{N}"
    cell_out.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).resolve().parent / "src"
    cmd = build_cmd(python_exe, app, N, cell_out, batch_size=batch_size)
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(
            cmd, cwd=str(src_dir), capture_output=True, text=True,
            timeout=TIMEOUT_S,
        )
        wall = time.perf_counter() - t0
        rc = cp.returncode
        err_tail = (cp.stderr or "").strip().splitlines()[-5:]
    except subprocess.TimeoutExpired as e:
        wall = time.perf_counter() - t0
        rc = -9
        err_tail = [f"timeout {TIMEOUT_S}s"]

    # Pull per_sweep_s + observables from summary.json if it exists.
    sj = cell_out / "summary.json"
    per_sweep_s = float("nan")
    total_wall_s = float("nan")
    re2_mean = float("nan")
    rg2_mean = float("nan")
    bond_min = float("nan")
    bond_max = float("nan")
    if sj.exists():
        try:
            d = json.loads(sj.read_text(encoding="utf-8"))
            timing = d.get("timing", {}) or {}
            obs = d.get("observables", {}) or {}
            total_wall_s = timing.get("total_wall_s") or timing.get("prod_wall_s") or float("nan")
            # per_sweep_s prefer prod-only / mean_per_sweep_s if present
            for key in ("mean_per_sweep_s", "per_sweep_s_prod", "per_sweep_s",
                        "prod_per_sweep_s"):
                if key in (timing or {}):
                    per_sweep_s = float(timing[key])
                    break
            if not math.isfinite(per_sweep_s):
                # derive from total / total_sweeps
                if total_wall_s and math.isfinite(total_wall_s):
                    per_sweep_s = total_wall_s / (EQ + PROD)
            re2_mean = obs.get("re2_mean", float("nan"))
            rg2_mean = obs.get("rg2_mean", float("nan"))
            bond_min = obs.get("bond_min", float("nan"))
            bond_max = obs.get("bond_max", float("nan"))
        except Exception as e:
            err_tail = err_tail + [f"summary parse: {e}"]
    return {
        "app": app,
        "N": N,
        "rc": rc,
        "wall_s_subprocess": round(wall, 3),
        "total_wall_s": total_wall_s if isinstance(total_wall_s, float) else float(total_wall_s),
        "per_sweep_s": per_sweep_s if isinstance(per_sweep_s, float) else float(per_sweep_s),
        "re2_mean": re2_mean if isinstance(re2_mean, float) else float(re2_mean),
        "rg2_mean": rg2_mean if isinstance(rg2_mean, float) else float(rg2_mean),
        "bond_min": bond_min if isinstance(bond_min, float) else float(bond_min),
        "bond_max": bond_max if isinstance(bond_max, float) else float(bond_max),
        "stderr_tail": " | ".join(err_tail),
        "summary_present": sj.exists(),
    }


PAIRS_FOR_MS_VS_CONV = [
    ("c1c_nb_cpu_single_core_conventional_mc_numba",     "c1m_nb_cpu_single_core_multistep_mc_numba"),
    ("c1c_pt_cpu_single_core_conventional_mc_py_torch",  "c1m_pt_cpu_single_core_multistep_mc_py_torch"),
    ("cnc_nb_cpu_multi_core_conventional_mc_numba",      "cnm_nb_cpu_multi_core_multistep_mc_numba"),
    ("cnc_pt_cpu_multi_core_conventional_mc_py_torch",   "cnm_pt_cpu_multi_core_multistep_mc_py_torch"),
    ("g1c_cc_gpu_single_thread_conventional_mc_cuda_c",  "g1m_cc_gpu_single_thread_multistep_mc_cuda_c"),
    ("g1c_pt_gpu_single_thread_conventional_mc_py_torch","g1m_pt_gpu_single_thread_multistep_mc_py_torch"),
    ("gnc_cc_gpu_multi_thread_conventional_mc_cuda_c",   "gnm_cc_gpu_multi_thread_multistep_mc_cuda_c"),
    ("gnc_pt_gpu_multi_thread_conventional_mc_py_torch", "gnm_pt_gpu_multi_thread_multistep_mc_py_torch"),
]

# CUDA-C should beat py_torch and numba *on the same hardware tier (single/multi)*.
# We compare GPU CUDA-C vs GPU py_torch directly. Numba comparison is CPU-only.
# Bug rule from /loop input: CUDA-C >= Numba/PyTorch counts as bug. We interpret
# this as: gpu_cuda_c per_sweep_s >= corresponding gpu_py_torch (same algo+tier).
CUDA_C_VS_PY_TORCH_PAIRS = [
    ("g1c_cc_gpu_single_thread_conventional_mc_cuda_c",  "g1c_pt_gpu_single_thread_conventional_mc_py_torch"),
    ("g1m_cc_gpu_single_thread_multistep_mc_cuda_c",     "g1m_pt_gpu_single_thread_multistep_mc_py_torch"),
    ("gnc_cc_gpu_multi_thread_conventional_mc_cuda_c",   "gnc_pt_gpu_multi_thread_conventional_mc_py_torch"),
    ("gnm_cc_gpu_multi_thread_multistep_mc_cuda_c",      "gnm_pt_gpu_multi_thread_multistep_mc_py_torch"),
]


def write_tsv(rows, path: Path, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def detect_bugs(timings_by_app_N):
    bugs = []
    crashes = []
    # Crash detection
    for (app, N), row in timings_by_app_N.items():
        if row["rc"] != 0:
            crashes.append((app, N, row["rc"], row["stderr_tail"]))
        else:
            ps = row.get("per_sweep_s", float("nan"))
            if not (isinstance(ps, float) and math.isfinite(ps)):
                crashes.append((app, N, "no per_sweep_s", row["stderr_tail"]))
            for key in ("re2_mean", "rg2_mean"):
                v = row.get(key, float("nan"))
                if isinstance(v, float) and not math.isfinite(v):
                    crashes.append((app, N, f"{key} non-finite", ""))

    # Multistep vs conventional
    for conv, mult in PAIRS_FOR_MS_VS_CONV:
        for N in N_VALUES:
            tc = timings_by_app_N.get((conv, N), {}).get("per_sweep_s", float("nan"))
            tm = timings_by_app_N.get((mult, N), {}).get("per_sweep_s", float("nan"))
            if math.isfinite(tc) and math.isfinite(tm):
                ratio = tc / tm if tm > 0 else float("inf")
                # Bug if multistep >= conventional (i.e., tm >= tc, ratio <= 1)
                if tm >= tc:
                    bugs.append({
                        "type": "multistep_not_faster",
                        "pair": f"{conv} vs {mult}",
                        "N": N,
                        "t_conv": tc, "t_mult": tm,
                        "speedup": ratio,
                        "note": "multistep MC per-sweep >= conventional (expected speedup_migacz>=2 on GPU)",
                    })

    # CUDA-C vs PyTorch (same algo + parallelism tier)
    for cuda_app, pt_app in CUDA_C_VS_PY_TORCH_PAIRS:
        for N in N_VALUES:
            tc = timings_by_app_N.get((cuda_app, N), {}).get("per_sweep_s", float("nan"))
            tp = timings_by_app_N.get((pt_app, N), {}).get("per_sweep_s", float("nan"))
            if math.isfinite(tc) and math.isfinite(tp):
                ratio = tp / tc if tc > 0 else float("inf")
                # Bug if cuda_c >= pytorch (cuda_c slower)
                if tc >= tp:
                    bugs.append({
                        "type": "cuda_c_not_faster_than_pytorch",
                        "pair": f"{cuda_app} vs {pt_app}",
                        "N": N,
                        "t_cuda_c": tc, "t_pt": tp,
                        "speedup_cuda_c": ratio,
                        "note": "CUDA-C per-sweep >= PyTorch on same tier (kernel should win)",
                    })

    # CUDA-C vs Numba: only meaningful pairing is GPU CUDA-C vs CPU Numba (cross-hardware).
    # Bug if GPU CUDA-C is slower than the analogous CPU Numba multistep app.
    CUDA_C_VS_NUMBA_PAIRS = [
        ("g1c_cc_gpu_single_thread_conventional_mc_cuda_c",  "c1c_nb_cpu_single_core_conventional_mc_numba"),
        ("g1m_cc_gpu_single_thread_multistep_mc_cuda_c",     "c1m_nb_cpu_single_core_multistep_mc_numba"),
        ("gnc_cc_gpu_multi_thread_conventional_mc_cuda_c",   "cnc_nb_cpu_multi_core_conventional_mc_numba"),
        ("gnm_cc_gpu_multi_thread_multistep_mc_cuda_c",      "cnm_nb_cpu_multi_core_multistep_mc_numba"),
    ]
    for cuda_app, nb_app in CUDA_C_VS_NUMBA_PAIRS:
        for N in N_VALUES:
            tc = timings_by_app_N.get((cuda_app, N), {}).get("per_sweep_s", float("nan"))
            tn = timings_by_app_N.get((nb_app, N), {}).get("per_sweep_s", float("nan"))
            if math.isfinite(tc) and math.isfinite(tn):
                if tc >= tn:
                    bugs.append({
                        "type": "cuda_c_not_faster_than_numba",
                        "pair": f"{cuda_app} vs {nb_app}",
                        "N": N,
                        "t_cuda_c": tc, "t_numba": tn,
                        "speedup_cuda_c": tn / tc if tc > 0 else float("inf"),
                        "note": "GPU CUDA-C >= CPU Numba per-sweep (GPU should win)",
                    })

    return bugs, crashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="_stress_iter1")
    p.add_argument("--apps", nargs="*", default=None,
                   help="optional subset of apps to run")
    p.add_argument("--batch_size", type=int, default=1,
                   help="--batch_size passed to multistep apps (no effect on conventional apps)")
    args = p.parse_args()

    root = Path(__file__).resolve().parent / args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    apps = args.apps if args.apps else APPS
    python_exe = sys.executable

    print(f"[stress] root={root}")
    print(f"[stress] {len(apps)} apps x {len(N_VALUES)} N = {len(apps)*len(N_VALUES)} cells")
    rows = []
    timings_by_app_N = {}
    t0 = time.perf_counter()
    for i, app in enumerate(apps, 1):
        for N in N_VALUES:
            print(f"  [{i:02d}/{len(apps)}] {app} N={N} ...", flush=True)
            r = run_one(app, N, root, python_exe, batch_size=args.batch_size)
            rows.append(r)
            timings_by_app_N[(app, N)] = r
            print(f"      rc={r['rc']} wall={r['wall_s_subprocess']:.2f}s "
                  f"per_sweep={r['per_sweep_s']:.4g}s "
                  f"re2={r.get('re2_mean','nan')} rg2={r.get('rg2_mean','nan')}")
    elapsed = time.perf_counter() - t0
    print(f"[stress] all cells done in {elapsed:.1f}s")

    # write timing TSV
    fields = ["app","N","rc","wall_s_subprocess","total_wall_s","per_sweep_s",
              "re2_mean","rg2_mean","bond_min","bond_max","summary_present","stderr_tail"]
    write_tsv(rows, root / "stress_timings.tsv", fields)

    bugs, crashes = detect_bugs(timings_by_app_N)
    bug_path = root / "stress_bugs.tsv"
    bug_fields = ["type","pair","N","t_conv","t_mult","t_cuda_c","t_pt","t_numba",
                  "speedup","speedup_cuda_c","note"]
    write_tsv(bugs, bug_path, bug_fields)
    crash_path = root / "stress_crashes.tsv"
    with crash_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["app","N","rc","stderr_tail"])
        for c in crashes:
            w.writerow(c)

    summary_path = root / "stress_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Stress test iteration\n")
        f.write(f"Cells: {len(rows)} | Wall: {elapsed:.1f}s\n")
        f.write(f"Crashes / non-finite: {len(crashes)}\n")
        f.write(f"Bugs flagged: {len(bugs)}\n\n")
        f.write("Bug summary by type:\n")
        from collections import Counter
        c = Counter(b["type"] for b in bugs)
        for k, v in c.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nFirst 20 bug rows:\n")
        for b in bugs[:20]:
            f.write(f"  {b}\n")
        f.write("\nCrashes:\n")
        for c in crashes[:20]:
            f.write(f"  {c}\n")
    print(f"[stress] wrote {summary_path}")
    print(f"[stress] bugs={len(bugs)} crashes={len(crashes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
