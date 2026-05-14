"""Pair-mode smoke harness for the 16 independent_apps backends.

Runs the 16 apps in 8 (CPU, GPU) pairs, one pair at a time, both apps in
the pair concurrent. After all pairs, audits each app's
summary.json + prod_observables.tsv + run.log against per-app gates and
flags cross-app physics outliers.

Usage:
    python -m independent_apps.run_pairs [--run_root <dir>]

Invocation matches verify_backends.py: each app is launched with
    python -m independent_apps.<app_name>.main <args>
from the parent of `src/` (so `src/` resolves as the package
`independent_apps`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from typing import Optional


PAIRS = [
    # (cpu_app, gpu_app)
    ("c1c_nb_cpu_single_core_conventional_mc_numba",    "g1c_cc_gpu_single_thread_conventional_mc_cuda_c"),
    ("c1c_pt_cpu_single_core_conventional_mc_py_torch", "g1c_pt_gpu_single_thread_conventional_mc_py_torch"),
    ("c1m_nb_cpu_single_core_multistep_mc_numba",       "g1m_cc_gpu_single_thread_multistep_mc_cuda_c"),
    ("c1m_pt_cpu_single_core_multistep_mc_py_torch",    "g1m_pt_gpu_single_thread_multistep_mc_py_torch"),
    ("cnc_nb_cpu_multi_core_conventional_mc_numba",     "gnc_cc_gpu_multi_thread_conventional_mc_cuda_c"),
    ("cnc_pt_cpu_multi_core_conventional_mc_py_torch",  "gnc_pt_gpu_multi_thread_conventional_mc_py_torch"),
    ("cnm_nb_cpu_multi_core_multistep_mc_numba",        "gnm_cc_gpu_multi_thread_multistep_mc_cuda_c"),
    ("cnm_pt_cpu_multi_core_multistep_mc_py_torch",     "gnm_pt_gpu_multi_thread_multistep_mc_py_torch"),
]


def _smoke_args(app_name: str, output_dir: str, bit_exact: bool = False) -> list[str]:
    eq_s = "50" if bit_exact else "300"
    prod_s = "50" if bit_exact else "300"
    args = [
        "--N", "25",
        "--n_chains", "8",
        "--phi", "0.10",
        "--eq_sweeps", eq_s,
        "--prod_sweeps", prod_s,
        "--seed", "42",
        "--init_method", "random_saw",
        "--residues_per_segment", "8",
        "--max_angle_hinge_pi", "0.5",
        "--traj_stride", "0",
        "--output_dir", output_dir,
        "--log_level", "INFO",
        "--log_file", os.path.join(output_dir, "run.log"),
    ]
    # Multistep batch size: 1 in bit-exact mode (degenerates to conventional MC),
    # 64 otherwise.
    if "multistep_mc" in app_name:
        args += ["--batch_size", "1" if bit_exact else "64"]
    # Per-app concurrency flags. Pinned to single-stream serialism in bit-exact
    # mode; otherwise pinned to 2 for parallelism smoke coverage.
    n_par = "1" if bit_exact else "2"
    if app_name.startswith("cpu_multi_core"):
        args += ["--threads", n_par]
        if app_name == "cnm_pt_cpu_multi_core_multistep_mc_py_torch":
            args += ["--workers", n_par]
    if app_name.startswith("gpu_multi_thread"):
        args += ["--streams", n_par]
    return args


def _launch(app_name: str, output_dir: str, src_dir: str,
            bit_exact: bool = False) -> subprocess.Popen:
    os.makedirs(output_dir, exist_ok=True)
    cmd = [sys.executable, "-m", f"{app_name}.main"] + \
          _smoke_args(app_name, output_dir, bit_exact=bit_exact)
    stdout_path = os.path.join(output_dir, "run.stdout")
    out_f = open(stdout_path, "w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=src_dir,
        stdout=out_f,
        stderr=subprocess.STDOUT,
    )


def _read_obs_tsv(path: str) -> dict:
    """Return {sweep: (Rg2, Re2, acc_hinge, acc_ntail, acc_ctail)}."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f, delimiter="\t")
        try:
            next(rdr)  # header
        except StopIteration:
            return {}
        for row in rdr:
            try:
                out[int(row[0])] = tuple(float(x) for x in row[1:6])
            except (ValueError, IndexError):
                continue
    return out


def _diff_streams(rows: list, run_root: str, tol: float = 1e-9) -> list:
    """Pairwise per-sweep observable diff across all 16 apps. Emits
    divergence_map.md and returns a list of divergent-pair records."""
    streams = {}
    for r in rows:
        app = r["app"]
        eq = _read_obs_tsv(os.path.join(run_root, app, "eq_observables.tsv"))
        prod = _read_obs_tsv(os.path.join(run_root, app, "prod_observables.tsv"))
        s = {}
        for sweep, vals in eq.items():
            s[("eq", sweep)] = vals
        for sweep, vals in prod.items():
            s[("prod", sweep)] = vals
        if s:
            streams[app] = s
    apps = sorted(streams.keys())
    diffs = []
    for i in range(len(apps)):
        for j in range(i + 1, len(apps)):
            a_i, a_j = apps[i], apps[j]
            s_i, s_j = streams[a_i], streams[a_j]
            common = sorted(set(s_i) & set(s_j),
                            key=lambda k: (0 if k[0] == "eq" else 1, k[1]))
            if not common:
                continue
            max_drg = 0.0
            max_dre = 0.0
            first_div = None
            for k in common:
                v_i = s_i[k]
                v_j = s_j[k]
                drg = abs(v_i[0] - v_j[0])
                dre = abs(v_i[1] - v_j[1])
                if drg > max_drg:
                    max_drg = drg
                if dre > max_dre:
                    max_dre = dre
                if first_div is None and (drg > tol or dre > tol):
                    first_div = k
            if first_div is not None:
                diffs.append({
                    "app_i": a_i, "app_j": a_j,
                    "first_phase": first_div[0],
                    "first_sweep": first_div[1],
                    "max_drg2": max_drg,
                    "max_dre2": max_dre,
                })
    diffs.sort(key=lambda d: (
        0 if d["first_phase"] == "eq" else 1,
        d["first_sweep"],
        -max(d["max_drg2"], d["max_dre2"]),
    ))
    out_path = os.path.join(run_root, "divergence_map.md")
    n_pairs = (len(apps) * (len(apps) - 1)) // 2
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Bit-exact divergence map\n\n")
        f.write(f"- Apps with observable streams: {len(apps)}\n")
        f.write(f"- Pairs compared: {n_pairs}\n")
        f.write(f"- Tolerance: {tol:.0e} (absolute, on Rg2 and Re2)\n")
        f.write(f"- Divergent pairs: {len(diffs)}\n\n")
        if not diffs:
            f.write("All app pairs identical within tolerance.\n")
        else:
            f.write("Sorted by earliest divergence, then by magnitude.\n\n")
            f.write("| app_i | app_j | first phase | first sweep | "
                    "max ΔRg² | max ΔRe² |\n")
            f.write("|---|---|---|---|---|---|\n")
            for d in diffs:
                f.write(f"| {d['app_i']} | {d['app_j']} | "
                        f"{d['first_phase']} | {d['first_sweep']} | "
                        f"{d['max_drg2']:.6e} | {d['max_dre2']:.6e} |\n")
    print(f"[run_pairs] wrote {out_path}", flush=True)
    return diffs


def _audit_app(app_name: str, output_dir: str, exit_code: int, wall_s: float) -> dict:
    """Return a row dict with gate pass/fail + key numbers."""
    row = {
        "app": app_name,
        "exit_code": exit_code,
        "wall_s": wall_s,
        "per_sweep_s": math.nan,
        "rg2_mean": math.nan,
        "re2_mean": math.nan,
        "ratio_re2_rg2": math.nan,
        "acc_hinge": math.nan,
        "acc_ntail": math.nan,
        "acc_ctail": math.nan,
        "bond_stretch_event": False,
        "traceback_in_log": False,
        "gate_status": "FAIL",
        "gate_reason": "",
    }
    if exit_code != 0:
        row["gate_reason"] = f"exit_code={exit_code}"
        return row
    summary_path = os.path.join(output_dir, "summary.json")
    if not os.path.exists(summary_path):
        row["gate_reason"] = "summary.json missing"
        return row
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception as e:
        row["gate_reason"] = f"summary.json unreadable: {e}"
        return row
    timing = s.get("timing", {})
    obs = s.get("observables", {})
    acc = s.get("acceptance", {})
    prod_per_s = float(timing.get("prod_sweeps_per_s", 0.0))
    row["per_sweep_s"] = 1.0 / prod_per_s if prod_per_s > 0 else math.nan
    row["rg2_mean"] = float(obs.get("rg2_mean", math.nan))
    row["re2_mean"] = float(obs.get("re2_mean", math.nan))
    if row["rg2_mean"] > 0:
        row["ratio_re2_rg2"] = row["re2_mean"] / row["rg2_mean"]
    row["acc_hinge"]  = float(acc.get("hinge",  math.nan))
    row["acc_ntail"]  = float(acc.get("n_tail", math.nan))
    row["acc_ctail"]  = float(acc.get("c_tail", math.nan))
    log_path = os.path.join(output_dir, "run.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            tail = f.read()
        row["bond_stretch_event"] = "[BOND-STRETCH]" in tail or "exceeds raise" in tail
        row["traceback_in_log"]   = "Traceback (most recent" in tail
    failures = []
    if not (row["per_sweep_s"] > 0 and math.isfinite(row["per_sweep_s"])):
        failures.append("per_sweep_s")
    if not (row["rg2_mean"] > 0 and math.isfinite(row["rg2_mean"])):
        failures.append("rg2_mean")
    if not (row["re2_mean"] > 0 and math.isfinite(row["re2_mean"])):
        failures.append("re2_mean")
    for k in ("acc_hinge", "acc_ntail", "acc_ctail"):
        v = row[k]
        if not (math.isfinite(v) and 0.05 <= v <= 0.95):
            failures.append(k)
    if row["bond_stretch_event"]:
        failures.append("bond_stretch")
    if row["traceback_in_log"]:
        failures.append("traceback")
    if not failures:
        row["gate_status"] = "PASS"
    else:
        row["gate_status"] = "FAIL"
        row["gate_reason"] = ",".join(failures)
    return row


def _human_seconds(s: float) -> str:
    return f"{s:.1f}s" if math.isfinite(s) else "n/a"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="pair-mode smoke harness")
    p.add_argument("--run_root", default="_pair_run",
                   help="output root (relative to repo root or absolute)")
    p.add_argument("--apps", default="",
                   help="comma-separated app-name allowlist; pairs with neither "
                        "side in the list are skipped; pairs with only one side "
                        "matching run that side alone")
    p.add_argument("--bit-exact", action="store_true", dest="bit_exact",
                   help="run all apps with --batch_size 1, --threads 1, "
                        "--workers 1, --streams 1 (homogeneous deterministic "
                        "mode); emit a pairwise divergence map after the run")
    args = p.parse_args(argv)

    src_dir = os.path.abspath(os.path.dirname(__file__))   # .../src
    repo_root = os.path.abspath(os.path.join(src_dir, ".."))  # repo root
    run_root = args.run_root
    if not os.path.isabs(run_root):
        run_root = os.path.join(repo_root, run_root)
    os.makedirs(run_root, exist_ok=True)

    allow = set(a.strip() for a in args.apps.split(",") if a.strip())
    print(f"[run_pairs] src_dir={src_dir}", flush=True)
    print(f"[run_pairs] run_root={run_root}", flush=True)
    if allow:
        print(f"[run_pairs] allowlist: {sorted(allow)}", flush=True)
    print(f"[run_pairs] launching {len(PAIRS)} pairs (CPU+GPU concurrent)", flush=True)

    rows = []
    for i, (cpu_app, gpu_app) in enumerate(PAIRS, 1):
        run_cpu = (not allow) or (cpu_app in allow)
        run_gpu = (not allow) or (gpu_app in allow)
        if not run_cpu and not run_gpu:
            print(f"[PAIR {i}/{len(PAIRS)}] skipped (neither side in allowlist)", flush=True)
            continue
        cpu_dir = os.path.join(run_root, cpu_app)
        gpu_dir = os.path.join(run_root, gpu_app)
        t0 = time.perf_counter()
        cpu_proc = _launch(cpu_app, cpu_dir, src_dir,
                           bit_exact=args.bit_exact) if run_cpu else None
        gpu_proc = _launch(gpu_app, gpu_dir, src_dir,
                           bit_exact=args.bit_exact) if run_gpu else None
        msg = f"[PAIR {i}/{len(PAIRS)}] launched:"
        if run_cpu: msg += f" cpu={cpu_app}"
        if run_gpu: msg += f" gpu={gpu_app}"
        print(msg, flush=True)
        if cpu_proc is not None:
            cpu_ec = cpu_proc.wait()
            cpu_wall = time.perf_counter() - t0
            cpu_row = _audit_app(cpu_app, cpu_dir, cpu_ec, cpu_wall)
            rows.append(cpu_row)
        if gpu_proc is not None:
            gpu_ec = gpu_proc.wait()
            gpu_wall = time.perf_counter() - t0
            gpu_row = _audit_app(gpu_app, gpu_dir, gpu_ec, gpu_wall)
            rows.append(gpu_row)
        for r in (cpu_row if run_cpu else None, gpu_row if run_gpu else None):
            if r is None:
                continue
            wall = r["wall_s"]
            print(f"[PAIR {i}/{len(PAIRS)}] {r['gate_status']:5} {r['app']:48} "
                  f"exit={r['exit_code']} wall={_human_seconds(wall)} "
                  f"R2={r['re2_mean']:.2f} Rg2={r['rg2_mean']:.2f} "
                  f"hinge={r['acc_hinge']:.3f} note={r['gate_reason'] or '-'}", flush=True)

    # Cross-app comparison
    re2_values = [r["re2_mean"] for r in rows if math.isfinite(r["re2_mean"])]
    rg2_values = [r["rg2_mean"] for r in rows if math.isfinite(r["rg2_mean"])]
    re2_med = statistics.median(re2_values) if re2_values else math.nan
    rg2_med = statistics.median(rg2_values) if rg2_values else math.nan
    for r in rows:
        if math.isfinite(r["re2_mean"]) and re2_med > 0:
            r["re2_dev_pct"] = 100.0 * (r["re2_mean"] - re2_med) / re2_med
        else:
            r["re2_dev_pct"] = math.nan
        if math.isfinite(r["rg2_mean"]) and rg2_med > 0:
            r["rg2_dev_pct"] = 100.0 * (r["rg2_mean"] - rg2_med) / rg2_med
        else:
            r["rg2_dev_pct"] = math.nan
        # outlier flag (independent of per-app gate)
        outlier = False
        if math.isfinite(r["re2_dev_pct"]) and abs(r["re2_dev_pct"]) > 20.0:
            outlier = True
        if math.isfinite(r["rg2_dev_pct"]) and abs(r["rg2_dev_pct"]) > 20.0:
            outlier = True
        if math.isfinite(r["ratio_re2_rg2"]) and not (3.0 <= r["ratio_re2_rg2"] <= 12.0):
            outlier = True
        r["outlier"] = outlier

    # Write comparison.tsv
    comp_path = os.path.join(run_root, "comparison.tsv")
    cols = ["app", "gate_status", "exit_code", "wall_s", "per_sweep_s",
            "re2_mean", "rg2_mean", "ratio_re2_rg2",
            "re2_dev_pct", "rg2_dev_pct", "outlier",
            "acc_hinge", "acc_ntail", "acc_ctail",
            "bond_stretch_event", "traceback_in_log", "gate_reason"]
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"[run_pairs] wrote {comp_path}", flush=True)

    # Write report.md
    report_path = os.path.join(run_root, "report.md")
    n_pass = sum(1 for r in rows if r["gate_status"] == "PASS")
    n_fail = len(rows) - n_pass
    n_outlier = sum(1 for r in rows if r["outlier"])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Pair-run report\n\n")
        f.write(f"- Apps run: {len(rows)}\n")
        f.write(f"- Per-app gate: {n_pass} PASS, {n_fail} FAIL\n")
        f.write(f"- Cross-app outliers: {n_outlier} (|deviation| > 20% of median, or ratio outside [3,12])\n")
        f.write(f"- Median R²={re2_med:.3f}, median Rg²={rg2_med:.3f}\n\n")
        f.write("## Per-app summary\n\n")
        f.write("| app | gate | exit | per_sweep_s | R² | Rg² | R²/Rg² | hinge | n_tail | c_tail | dev_R²(%) | dev_Rg²(%) | outlier | reason |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['app']} | {r['gate_status']} | {r['exit_code']} | "
                    f"{r['per_sweep_s']:.4f} | {r['re2_mean']:.2f} | {r['rg2_mean']:.2f} | "
                    f"{r['ratio_re2_rg2']:.2f} | {r['acc_hinge']:.3f} | "
                    f"{r['acc_ntail']:.3f} | {r['acc_ctail']:.3f} | "
                    f"{r['re2_dev_pct']:+.1f} | {r['rg2_dev_pct']:+.1f} | "
                    f"{'YES' if r['outlier'] else '-'} | {r['gate_reason'] or '-'} |\n")
    print(f"[run_pairs] wrote {report_path}", flush=True)
    print(f"[run_pairs] summary: {n_pass}/{len(rows)} PASS  {n_outlier} outliers", flush=True)

    n_div = 0
    if args.bit_exact:
        diffs = _diff_streams(rows, run_root)
        n_div = len(diffs)
        print(f"[run_pairs] bit-exact divergent pairs: {n_div}", flush=True)

    return 0 if (n_fail == 0 and n_div == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
