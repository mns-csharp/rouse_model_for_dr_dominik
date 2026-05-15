"""Measurement gate for the g1m_ccx streamed CUDA-C multistep optimization.

Focused K-sweep: the new optimized multistep app `g1m_ccx` vs the two
conventional CUDA-C targets it must beat — `g1c_cc` (single-thread) and
`gnc_cc` (multi-thread). No full benchmark matrix; just the gate question:
does g1m_ccx's throughput cross g1c_cc / gnc_cc at any K?

eq=prod=100 (no scale-down); N=50; K sweeps the multistep batch B across the
Migacz inefficient->efficient transition. Each cell is an isolated subprocess.

Run from repo root:  python _gate_ccx.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from _bench_runner import run_subprocess, detect_status

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"

N = 50
K_VALUES = [20, 64, 128, 192, 256]
EQ = 100
PROD = 100
SEED = 42
PHI = 0.01
PER_CELL_TIMEOUT_S = 6000

CHALLENGERS = [
    "g1m_ccx_gpu_single_thread_multistep_mc_cuda_c_streamed",
    "gnm_ccx_gpu_multi_thread_multistep_mc_cuda_c_streamed",
]
TARGETS = [
    "g1c_cc_gpu_single_thread_conventional_mc_cuda_c",
    "gnc_cc_gpu_multi_thread_conventional_mc_cuda_c",
]
APPS = CHALLENGERS + TARGETS

OUT = REPO / "_gate_ccx_out"
RUNS = OUT / "_runs"


def short(app: str) -> str:
    return "_".join(app.split("_")[:2])


def run_cell(app: str, K: int) -> dict:
    out_dir = RUNS / f"{app}_N{N}_K{K}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", f"{app}.main",
        "--N", str(N), "--n_chains", str(K),
        "--eq_sweeps", str(EQ), "--prod_sweeps", str(PROD),
        "--seed", str(SEED), "--phi", str(PHI),
        "--output_dir", str(out_dir),
        "--traj_stride", "0",
        "--heartbeat_interval", "0",
        "--log_level", "WARNING",
    ]
    res = run_subprocess(cmd, cwd=SRC, timeout=PER_CELL_TIMEOUT_S)
    status = detect_status(res["rc"], res["timed_out"], res["stderr"], res["stdout"])

    summary = None
    sp = out_dir / "summary.json"
    if sp.exists():
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            summary = None

    prod_wall = prod_sps = re2 = rg2 = None
    if summary:
        timing = summary.get("timing", {}) or {}
        observ = summary.get("observables", {}) or {}
        prod_wall = timing.get("prod_wall_s") or summary.get("prod_wall_s")
        prod_sps = timing.get("prod_sweeps_per_s") or summary.get("prod_sweeps_per_s")
        if prod_sps is None and prod_wall:
            prod_sps = PROD / max(prod_wall, 1e-12)
        re2 = observ.get("re2_mean") or summary.get("Re2_mean")
        rg2 = observ.get("rg2_mean") or summary.get("Rg2_mean")

    return {
        "status": status, "prod_wall_s": prod_wall,
        "prod_sweeps_per_s": prod_sps, "Re2_mean": re2, "Rg2_mean": rg2,
        "stderr_tail": (res["stderr"] or "")[-400:],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"=== CCX GATE === N={N} K={K_VALUES} eq=prod={PROD} phi={PHI}", flush=True)

    records: dict = {a: {} for a in APPS}
    for K in K_VALUES:
        for app in APPS:
            print(f"[run] {short(app)} N={N} K={K} ...", flush=True)
            cell = run_cell(app, K)
            records[app][K] = cell
            print(f"      status={cell['status']} "
                  f"sps={cell['prod_sweeps_per_s']} "
                  f"Re2={cell['Re2_mean']} Rg2={cell['Rg2_mean']}", flush=True)
            if cell["status"] != "ok":
                print(f"      stderr: {cell['stderr_tail']}", flush=True)
    (OUT / "gate_records.json").write_text(
        json.dumps(records, indent=2, default=str), encoding="utf-8")

    # --- gate verdict -----------------------------------------------------
    print("\n=== GATE TABLE (prod sweeps/s) ===", flush=True)
    print("K".ljust(6) + "".join(short(a).ljust(14) for a in APPS), flush=True)
    for K in K_VALUES:
        row = str(K).ljust(6)
        for a in APPS:
            v = records[a][K].get("prod_sweeps_per_s")
            row += (f"{v:.3f}" if v else "—").ljust(14)
        print(row, flush=True)

    all_pass = True
    for ch in CHALLENGERS:
        crossed = []
        for K in K_VALUES:
            c = records[ch][K].get("prod_sweeps_per_s")
            tgt = [records[t][K].get("prod_sweeps_per_s") for t in TARGETS]
            if c and all(tgt) and all(c >= t for t in tgt):
                crossed.append(K)
        print("", flush=True)
        if crossed:
            print(f"[GATE PASS] {short(ch)} beats BOTH {short(TARGETS[0])} and "
                  f"{short(TARGETS[1])} at K={crossed}.", flush=True)
        else:
            print(f"[GATE FAIL] {short(ch)} does not beat both conventional "
                  f"targets at any tested K.", flush=True)
            all_pass = False
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
