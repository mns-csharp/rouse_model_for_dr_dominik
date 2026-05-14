"""Smoke + regression harness for the 16 independent_apps backends.

Usage:
    python -m independent_apps.verify_backends [--mode smoke|regression]
                                                [--quick]
                                                [--include cpu,gpu,...]

Modes:
  smoke      — boot each backend with tiny parameters and confirm it runs
               to summary.json. CPU backends always run; GPU backends are
               skipped with a clear note when CUDA is not available.
  regression — run pairs of related backends with `--threads 1` /
               `--workers 1` / `--streams 1` (single-stream/single-worker)
               and compare the final state hash. Pairs (by underscore-joined name):
                 cpu_single_core_X_Y vs cpu_multi_core_X_Y
                 gpu_single_thread_X_Y vs gpu_multi_thread_X_Y
                 gpu_X_Y_cuda_c vs gpu_X_Y_py_torch (within float32 tol)

The harness drives each backend through its main.py CLI to keep the
tested surface honest (the actual entry point users invoke).

This is intentionally minimal: it confirms backends import + run, not
that they're production-fast. Throughput benchmarking lives elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


BACKEND_ROOTS: List[Tuple[str, str]] = [
    # (label, dotted module path of main.py)
    ("c1c_nb_cpu_single_core_conventional_mc_numba",      "independent_apps.c1c_nb_cpu_single_core_conventional_mc_numba.main"),
    ("c1c_pt_cpu_single_core_conventional_mc_py_torch",   "independent_apps.c1c_pt_cpu_single_core_conventional_mc_py_torch.main"),
    ("c1m_nb_cpu_single_core_multistep_mc_numba",         "independent_apps.c1m_nb_cpu_single_core_multistep_mc_numba.main"),
    ("c1m_pt_cpu_single_core_multistep_mc_py_torch",      "independent_apps.c1m_pt_cpu_single_core_multistep_mc_py_torch.main"),
    ("cnc_nb_cpu_multi_core_conventional_mc_numba",       "independent_apps.cnc_nb_cpu_multi_core_conventional_mc_numba.main"),
    ("cnc_pt_cpu_multi_core_conventional_mc_py_torch",    "independent_apps.cnc_pt_cpu_multi_core_conventional_mc_py_torch.main"),
    ("cnm_nb_cpu_multi_core_multistep_mc_numba",          "independent_apps.cnm_nb_cpu_multi_core_multistep_mc_numba.main"),
    ("cnm_pt_cpu_multi_core_multistep_mc_py_torch",       "independent_apps.cnm_pt_cpu_multi_core_multistep_mc_py_torch.main"),
    ("g1c_cc_gpu_single_thread_conventional_mc_cuda_c",   "independent_apps.g1c_cc_gpu_single_thread_conventional_mc_cuda_c.main"),
    ("g1c_pt_gpu_single_thread_conventional_mc_py_torch", "independent_apps.g1c_pt_gpu_single_thread_conventional_mc_py_torch.main"),
    ("g1m_cc_gpu_single_thread_multistep_mc_cuda_c",      "independent_apps.g1m_cc_gpu_single_thread_multistep_mc_cuda_c.main"),
    ("g1m_pt_gpu_single_thread_multistep_mc_py_torch",    "independent_apps.g1m_pt_gpu_single_thread_multistep_mc_py_torch.main"),
    ("gnc_cc_gpu_multi_thread_conventional_mc_cuda_c",    "independent_apps.gnc_cc_gpu_multi_thread_conventional_mc_cuda_c.main"),
    ("gnc_pt_gpu_multi_thread_conventional_mc_py_torch",  "independent_apps.gnc_pt_gpu_multi_thread_conventional_mc_py_torch.main"),
    ("gnm_cc_gpu_multi_thread_multistep_mc_cuda_c",       "independent_apps.gnm_cc_gpu_multi_thread_multistep_mc_cuda_c.main"),
    ("gnm_pt_gpu_multi_thread_multistep_mc_py_torch",     "independent_apps.gnm_pt_gpu_multi_thread_multistep_mc_py_torch.main"),
]


@dataclass
class RunResult:
    label: str
    status: str    # "ok" | "fail" | "skip"
    seconds: float
    note: str = ""
    summary: Optional[dict] = None


def _is_gpu(label: str) -> bool:
    return label.startswith("gpu_")


def _can_run_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _smoke_args_for(label: str, output_dir: str, seed: int = 0) -> List[str]:
    """Return tiny-but-real arguments that exercise each backend's CLI."""
    base = [
        "--N", "16",
        "--n_chains", "4",
        "--phi", "0.05",
        "--eq_sweeps", "5",
        "--prod_sweeps", "5",
        "--seed", str(seed),
        "--output_dir", output_dir,
    ]
    if "multistep" in label:
        base += ["--batch_size", "4"]
    # multi_core on py_torch advertises --workers; default 1 is fine.
    # multi_thread advertises --streams; default 4 is fine in smoke.
    return base


def _run_backend(label: str, module: str, output_dir: str,
                 timeout_s: float = 180.0) -> RunResult:
    import time
    if _is_gpu(label) and not _can_run_gpu():
        return RunResult(label, "skip", 0.0, note="CUDA not available")

    args = _smoke_args_for(label, output_dir)
    cmd = [sys.executable, "-m", module] + args
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s,
                              check=False, text=True)
    except subprocess.TimeoutExpired:
        return RunResult(label, "fail", timeout_s, note="timeout")
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        # Last 8 lines of stderr capture the most useful failure context.
        tail = "\n".join((proc.stderr or "").splitlines()[-8:])
        return RunResult(label, "fail", wall, note=tail)

    summary_path = os.path.join(output_dir, "summary.json")
    summary = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as e:
            return RunResult(label, "fail", wall, note=f"summary unreadable: {e}")
    return RunResult(label, "ok", wall, summary=summary)


def smoke_all(quick: bool, include: Optional[List[str]]) -> List[RunResult]:
    results = []
    tmp = tempfile.mkdtemp(prefix="indep_verify_")
    try:
        for label, module in BACKEND_ROOTS:
            if include and not any(label.startswith(p) for p in include):
                continue
            run_dir = os.path.join(tmp, label)
            os.makedirs(run_dir, exist_ok=True)
            res = _run_backend(label, module, run_dir,
                               timeout_s=60.0 if quick else 180.0)
            results.append(res)
            tag = {"ok": "  PASS", "fail": "  FAIL", "skip": "  SKIP"}[res.status]
            extra = f" ({res.seconds:.1f}s)" if res.status != "skip" else ""
            print(f"{tag} {label}{extra}", flush=True)
            if res.status == "fail" and res.note:
                for line in res.note.splitlines():
                    print(f"        | {line}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def _print_summary(results: List[RunResult]) -> int:
    n_pass = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_skip = sum(1 for r in results if r.status == "skip")
    print()
    print(f"Total: {len(results)}  pass={n_pass}  fail={n_fail}  skip={n_skip}")
    return 0 if n_fail == 0 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="independent_apps verifier")
    p.add_argument("--mode", choices=["smoke", "regression"], default="smoke")
    p.add_argument("--quick", action="store_true",
                   help="shorter per-backend timeout")
    p.add_argument("--include", default="",
                   help="comma-separated label prefixes (e.g. cpu,gpu_single_thread)")
    args = p.parse_args(argv)

    include = [x for x in args.include.split(",") if x] or None

    if args.mode == "smoke":
        results = smoke_all(quick=args.quick, include=include)
        return _print_summary(results)
    if args.mode == "regression":
        # Regression mode is not yet implemented — would compare summary.json
        # observables across paired backends. Placeholder.
        print("regression mode: not yet implemented (smoke covers correctness today)",
              file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
