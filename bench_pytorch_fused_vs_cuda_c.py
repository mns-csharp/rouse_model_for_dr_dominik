"""Bench: PyTorch-fused (4) vs CUDA-C (4) at the canonical Gate-3 cell.

Cell: N=100, K=100, phi=0.10, eq=5, prod=10, residues_per_segment=8.
Multistep apps: --batch_size 100. Multi-thread apps: --streams 4.
PyTorch-fused apps: ROUSE_TORCH_COMPILE_MODE=inductor in subprocess env.
Per-app timeout: 300 s (generous for NVRTC JIT on first CUDA-C invocation).

Outputs:
  _bench_pytorch_fused_vs_cuda_c/<app>/summary.json
  _bench_pytorch_fused_vs_cuda_c/results.tsv
  _bench_pytorch_fused_vs_cuda_c/comparison.tsv

Headline: per-sweep time table sorted ascending, plus a 4-row (algorithm ×
threading) PT-fused vs CUDA-C ratio table.
"""
from __future__ import annotations
import csv, json, math, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "_bench_pytorch_fused_vs_cuda_c"
OUT.mkdir(parents=True, exist_ok=True)

PT_FUSED_APPS = [
    "g1c_ptg_gpu_single_thread_conventional_mc_py_torch_gpu_fused",
    "gnc_ptg_gpu_multi_thread_conventional_mc_py_torch_gpu_fused",
    "g1m_ptg_gpu_single_thread_multistep_mc_py_torch_gpu_fused",
    "gnm_ptg_gpu_multi_thread_multistep_mc_py_torch_gpu_fused",
]
CUDA_C_APPS = [
    "g1c_cc_gpu_single_thread_conventional_mc_cuda_c",
    "gnc_cc_gpu_multi_thread_conventional_mc_cuda_c",
    "g1m_cc_gpu_single_thread_multistep_mc_cuda_c",
    "gnm_cc_gpu_multi_thread_multistep_mc_cuda_c",
]
APPS = PT_FUSED_APPS + CUDA_C_APPS

N = 100
K = 100
EQ = 5
PROD = 10
PHI = 0.10
SEED = 42
BATCH_SIZE_MULTISTEP = 100
N_STREAMS = 4
TIMEOUT_S = 300

# Reference: best CPU-Numba per-sweep at N=100 K=100, eq=5 prod=10, from
# previous bench (c1c_nb_cpu_single_core_conventional_mc_numba):
CPU_NUMBA_REF_PER_SWEEP_S = 8.89


def is_pt_fused(app: str) -> bool: return "py_torch_gpu_fused" in app
def is_multistep(app: str) -> bool: return "multistep" in app
def is_multi_thread(app: str) -> bool: return "multi_thread" in app


def build_cmd(app: str, cell_out: Path) -> list[str]:
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


def run_one(app: str) -> dict:
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
            err_tail = (err_tail + f" | parse: {e}")[-400:]
    return {
        "app": app,
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


def _algo_thread_key(app: str) -> str:
    algo = "multistep" if is_multistep(app) else "conventional"
    thr = "multi" if is_multi_thread(app) else "single"
    return f"{algo}_{thr}"


def main():
    rows: list[dict] = []
    t_start = time.perf_counter()
    n_total = len(APPS)
    for i, app in enumerate(APPS, 1):
        tag = f"[{i:02d}/{n_total}] app={app}"
        print(f"  {tag} ...", flush=True)
        r = run_one(app)
        rows.append(r)
        per = (f"{r['per_sweep_s']:.4g}s" if math.isfinite(r['per_sweep_s'])
               else "n/a")
        print(f"    -> status={r['status']} rc={r['rc']} "
              f"wall={r['wall_s_subprocess']:.1f}s per_sweep={per} "
              f"re2={r['re2_mean']:.1f}", flush=True)
    total_wall = time.perf_counter() - t_start
    print(f"\n[done] {n_total} apps in {total_wall:.1f}s")

    results_path = OUT / "results.tsv"
    write_tsv(rows, results_path, [
        "app", "rc", "wall_s_subprocess",
        "per_sweep_s", "re2_mean", "rg2_mean", "status", "err_tail",
    ])
    print(f"[wrote] {results_path}")

    # Build PT-fused vs CUDA-C comparison rows.
    by_key: dict[str, dict[str, dict]] = {}
    for r in rows:
        k = _algo_thread_key(r["app"])
        backend = "pt_fused" if is_pt_fused(r["app"]) else "cuda_c"
        by_key.setdefault(k, {})[backend] = r
    comp_rows = []
    for key in ("conventional_single", "conventional_multi",
                "multistep_single", "multistep_multi"):
        pair = by_key.get(key, {})
        pt = pair.get("pt_fused", {})
        cc = pair.get("cuda_c", {})
        pt_t = pt.get("per_sweep_s", float("nan"))
        cc_t = cc.get("per_sweep_s", float("nan"))
        ratio = (pt_t / cc_t) if (math.isfinite(pt_t) and math.isfinite(cc_t)
                                   and cc_t > 0) else float("nan")
        cuda_wins = math.isfinite(ratio) and ratio > 1.0
        comp_rows.append({
            "variant": key,
            "pt_fused_app": pt.get("app", "n/a"),
            "pt_fused_s_per_sweep": pt_t if math.isfinite(pt_t) else "n/a",
            "cuda_c_app": cc.get("app", "n/a"),
            "cuda_c_s_per_sweep": cc_t if math.isfinite(cc_t) else "n/a",
            "pt_over_cuda_ratio": ratio if math.isfinite(ratio) else "n/a",
            "cuda_c_wins": bool(cuda_wins),
        })
    comp_path = OUT / "comparison.tsv"
    write_tsv(comp_rows, comp_path, [
        "variant", "pt_fused_app", "pt_fused_s_per_sweep",
        "cuda_c_app", "cuda_c_s_per_sweep",
        "pt_over_cuda_ratio", "cuda_c_wins",
    ])
    print(f"[wrote] {comp_path}")

    print()
    print("=" * 88)
    print(f" HEADLINE TABLE — N={N} K={K} eq={EQ} prod={PROD} (sorted by per_sweep_s ascending) ")
    print("=" * 88)
    finite_rows = sorted(
        (r for r in rows if math.isfinite(r["per_sweep_s"])),
        key=lambda r: r["per_sweep_s"],
    )
    failed = [r for r in rows if not math.isfinite(r["per_sweep_s"])]
    best_pt = min(
        (r["per_sweep_s"] for r in finite_rows if is_pt_fused(r["app"])),
        default=float("nan"),
    )
    print(f"  {'rank':>4}  {'app':<60} {'s/sweep':>10}  {'×CPU-Numba':>11}  {'×best-PT':>9}")
    for rank, r in enumerate(finite_rows, 1):
        vs_cpu = CPU_NUMBA_REF_PER_SWEEP_S / r["per_sweep_s"]
        vs_pt = (best_pt / r["per_sweep_s"]) if math.isfinite(best_pt) else float("nan")
        print(f"  {rank:>4}  {r['app']:<60} {r['per_sweep_s']:>10.4g}  "
              f"{vs_cpu:>10.2f}×  "
              f"{vs_pt:>8.2f}×" if math.isfinite(vs_pt) else "      n/a")
    if failed:
        print("  --- failed/timeout ---")
        for r in failed:
            print(f"        {r['app']:<60} status={r['status']} err={r['err_tail'][:80]}")

    print()
    print("=" * 88)
    print(" PT-FUSED vs CUDA-C (per algorithm × threading) ")
    print("=" * 88)
    for r in comp_rows:
        pt_s = r["pt_fused_s_per_sweep"]
        cc_s = r["cuda_c_s_per_sweep"]
        ratio = r["pt_over_cuda_ratio"]
        pt_str = f"{pt_s:.4g}s" if isinstance(pt_s, float) else "n/a"
        cc_str = f"{cc_s:.4g}s" if isinstance(cc_s, float) else "n/a"
        ratio_str = f"{ratio:.2f}× PT/CUDA" if isinstance(ratio, float) else "n/a"
        winner = "CUDA-C" if r["cuda_c_wins"] else ("PT-fused"
                  if isinstance(ratio, float) and ratio < 1.0 else "n/a")
        print(f"  {r['variant']:>20}: PT={pt_str:>10}  CUDA={cc_str:>10}  "
              f"ratio={ratio_str:>16}  winner={winner}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
