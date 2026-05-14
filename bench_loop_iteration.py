"""Single iteration of the /loop benchmark workflow.

Picks the next unbenched app from src/, benchmarks at NK in {250, 1000, 10000}
(N in {25, 100, 1000}, K=10), 100 sweeps each (eq=10 + prod=90). Writes one
JSON file per app to benchmark_data/<app>.json. When all apps are done, builds
the four backend tables and four ratio plots.

Detects OOM / timeout / non-zero exit per (app, size) and records as such so
the loop can move on instead of stalling.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\git\rouse_model_python_independent_apps")
SRC = REPO_ROOT / "src"
BENCH_DIR = REPO_ROOT / "benchmark_data"
RUNS_DIR = BENCH_DIR / "_runs"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# (N, K) sizes
SIZES = [(25, 10), (100, 10), (1000, 10)]
EQ_SWEEPS = 10
PROD_SWEEPS = 90
SEED = 42

# Hard wall-clock timeout per (app, size).
PER_SIZE_TIMEOUT_S = 1200

BASELINE_APP = "c1c_nb_cpu_single_core_conventional_mc_numba"


def discover_apps() -> list[str]:
    apps = []
    for child in sorted(SRC.iterdir()):
        if not child.is_dir():
            continue
        if not re.match(r"^[cg][1n][cm]_[a-z]+_", child.name):
            continue
        if not (child / "main.py").exists():
            continue
        apps.append(child.name)
    return apps


def already_benched(app: str) -> bool:
    return (BENCH_DIR / f"{app}.json").exists()


def pick_next_app(apps: list[str]) -> str | None:
    # Ensure baseline goes first.
    if not already_benched(BASELINE_APP) and BASELINE_APP in apps:
        return BASELINE_APP
    for app in apps:
        if not already_benched(app):
            return app
    return None


def detect_failure_kind(stderr: str, stdout: str, returncode: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    text = (stderr or "") + "\n" + (stdout or "")
    low = text.lower()
    if "out of memory" in low or "cuda_error_out_of_memory" in low or "cudaerrormemoryallocation" in low:
        return "oom"
    if "cuda not available" in low:
        return "no_cuda"
    if returncode != 0:
        return "error"
    return "ok"


def run_one(app: str, N: int, K: int) -> dict:
    """Run app at (N, K). Return result dict with timings + status."""
    out_dir = RUNS_DIR / f"{app}_N{N}_K{K}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", f"{app}.main",
        "--N", str(N),
        "--n_chains", str(K),
        "--eq_sweeps", str(EQ_SWEEPS),
        "--prod_sweeps", str(PROD_SWEEPS),
        "--seed", str(SEED),
        "--output_dir", str(out_dir),
        "--traj_stride", "0",
        "--heartbeat_interval", "0",
        "--log_level", "WARNING",
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=SRC,
            capture_output=True,
            text=True,
            timeout=PER_SIZE_TIMEOUT_S,
            env=env,
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc = -1
        stdout = (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or ""))
        stderr = (e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or ""))
    wall = time.time() - t0

    status = detect_failure_kind(stderr, stdout, rc, timed_out)

    summary = None
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = None

    eq_wall = None
    prod_wall = None
    prod_sps = None
    re2 = None
    rg2 = None
    if summary:
        eq_wall = summary.get("eq_wall_s") or (summary.get("eq", {}) or {}).get("wall_s")
        prod_wall = summary.get("prod_wall_s") or (summary.get("prod", {}) or {}).get("wall_s")
        prod_sps = summary.get("prod_sweeps_per_s")
        if prod_sps is None and prod_wall and (summary.get("prod_n_sweeps") or PROD_SWEEPS):
            prod_sps = (summary.get("prod_n_sweeps") or PROD_SWEEPS) / max(prod_wall, 1e-12)
        re2 = summary.get("Re2_mean") or (summary.get("prod", {}) or {}).get("Re2_mean")
        rg2 = summary.get("Rg2_mean") or (summary.get("prod", {}) or {}).get("Rg2_mean")

    err_tail = (stderr or "")[-1500:]
    return {
        "N": N,
        "K": K,
        "NK": N * K,
        "status": status,
        "returncode": rc,
        "wall_total_s": wall,
        "eq_wall_s": eq_wall,
        "prod_wall_s": prod_wall,
        "prod_sweeps_per_s": prod_sps,
        "per_sweep_s": (prod_wall / PROD_SWEEPS) if prod_wall else None,
        "Re2_mean": re2,
        "Rg2_mean": rg2,
        "err_tail": err_tail,
        "cmd": " ".join(cmd),
    }


def bug_check(results_by_size: dict) -> list[str]:
    """Return list of detected anomalies. Empty list = clean."""
    bugs = []
    for size_key, res in results_by_size.items():
        if res["status"] not in ("ok", "oom", "timeout"):
            bugs.append(f"{size_key}: status={res['status']} rc={res['returncode']}")
        if res["status"] == "ok":
            if res["prod_wall_s"] is None or res["prod_wall_s"] <= 0:
                bugs.append(f"{size_key}: ok but missing prod wall time")
            re2 = res.get("Re2_mean")
            if re2 is not None:
                try:
                    v = float(re2)
                    if not (v == v) or v <= 0.0:
                        bugs.append(f"{size_key}: Re2_mean non-finite or <=0 ({re2})")
                except Exception:
                    bugs.append(f"{size_key}: Re2_mean unparseable ({re2!r})")
    return bugs


def bench_app(app: str) -> dict:
    results = {}
    for N, K in SIZES:
        key = f"N{N}_K{K}"
        print(f"[bench] {app} {key} ...", flush=True)
        results[key] = run_one(app, N, K)
        s = results[key]["status"]
        wt = results[key]["wall_total_s"]
        print(f"[bench] {app} {key} -> status={s} wall={wt:.1f}s", flush=True)
        # If we OOM/timeout at small NK, larger NK will fare no better — skip.
        if s in ("oom", "timeout") and (N, K) != SIZES[-1]:
            for N2, K2 in SIZES[SIZES.index((N, K)) + 1:]:
                k2 = f"N{N2}_K{K2}"
                results[k2] = {
                    "N": N2, "K": K2, "NK": N2 * K2,
                    "status": "skipped_after_" + s,
                    "returncode": None,
                    "wall_total_s": None, "eq_wall_s": None, "prod_wall_s": None,
                    "prod_sweeps_per_s": None, "per_sweep_s": None,
                    "Re2_mean": None, "Rg2_mean": None,
                    "err_tail": "", "cmd": "",
                }
            break
    return results


def compute_ratios(app: str, results: dict, baseline: dict | None) -> dict:
    """Compute throughput / speedup / scaling / wall-clock ratios.

    - throughput[size] = prod_sweeps_per_s
    - speedup[size]    = this.throughput / baseline.throughput (same size)
    - wall_ratio[size] = baseline.wall_total / this.wall_total (same size)
    - scaling[size]    = this.per_sweep_s[size] / this.per_sweep_s[smallest NK]
                         (for this app only; reveals NK scaling exponent)
    """
    ratios = {}
    sizes_ok = [k for k, r in results.items() if r["status"] == "ok" and r.get("per_sweep_s")]
    base_size = sizes_ok[0] if sizes_ok else None
    base_per_sweep = results[base_size]["per_sweep_s"] if base_size else None

    for size_key, r in results.items():
        d = {}
        if r["status"] == "ok":
            d["throughput_sps"] = r["prod_sweeps_per_s"]
            d["per_sweep_s"] = r["per_sweep_s"]
            d["wall_total_s"] = r["wall_total_s"]
            if base_per_sweep:
                d["scaling_vs_smallest"] = r["per_sweep_s"] / base_per_sweep
            if baseline and baseline.get("sizes", {}).get(size_key, {}).get("status") == "ok":
                b = baseline["sizes"][size_key]
                if b.get("prod_sweeps_per_s") and r.get("prod_sweeps_per_s"):
                    d["speedup_vs_baseline"] = r["prod_sweeps_per_s"] / b["prod_sweeps_per_s"]
                if b.get("wall_total_s") and r.get("wall_total_s"):
                    d["wall_ratio_baseline_over_this"] = b["wall_total_s"] / r["wall_total_s"]
        else:
            d["status"] = r["status"]
        ratios[size_key] = d
    return ratios


def load_app_record(app: str) -> dict | None:
    f = BENCH_DIR / f"{app}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_app_record(app: str, record: dict) -> None:
    f = BENCH_DIR / f"{app}.json"
    f.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def app_backend_tags(app: str) -> tuple[str, str, str]:
    """Return (hw, framework, algo) tags for grouping into the four tables."""
    code = app.split("_")[0]   # e.g. "g1c", "c1c", "gnm", "g1m"
    backend_code = app.split("_")[1]  # e.g. "nb", "pt", "cc", "ptg", "ptgcl", "cccl"
    hw = "GPU" if code[0] == "g" else "CPU"
    # Framework grouping:
    #   nb       -> Numba
    #   pt, ptc, ptf, ptg, ptgcl -> PyTorch
    #   cc, cccl -> CUDA-C
    if backend_code == "nb":
        fw = "Numba"
    elif backend_code in ("cc", "cccl"):
        fw = "CUDA-C"
    else:
        fw = "PyTorch"
    algo = "conventional" if code[2] == "c" else "multistep"
    return hw, fw, algo


def build_aggregates() -> None:
    """Build the four tables + four ratio plots. Called when all apps benched."""
    apps = discover_apps()
    records = {a: load_app_record(a) for a in apps}
    records = {a: r for a, r in records.items() if r is not None}

    groups = {
        ("CPU", "Numba"):   [],
        ("CPU", "PyTorch"): [],
        ("GPU", "CUDA-C"):  [],
        ("GPU", "PyTorch"): [],
    }
    for a, r in records.items():
        hw, fw, _ = app_backend_tags(a)
        key = (hw, fw)
        if key in groups:
            groups[key].append((a, r))

    # Write text tables.
    tables_path = BENCH_DIR / "tables.md"
    lines = []
    lines.append(f"# Benchmark tables (N in {{25,100,1000}}, K=10, prod=90 sweeps)")
    lines.append("")
    lines.append(f"Baseline for speedup/wall-ratio: `{BASELINE_APP}` at the same (N,K).")
    lines.append("")
    size_keys = [f"N{N}_K{K}" for N, K in SIZES]
    for (hw, fw), entries in groups.items():
        if not entries:
            continue
        lines.append(f"## {{{hw}, {fw}}}")
        lines.append("")
        hdr = "| App | NK | throughput (sw/s) | speedup × | scaling × | wall (s) |"
        sep = "|---|---|---|---|---|---|"
        lines.append(hdr)
        lines.append(sep)
        for app, rec in sorted(entries):
            sizes = rec.get("sizes", {})
            ratios = rec.get("ratios", {})
            for sk in size_keys:
                s = sizes.get(sk, {})
                rt = ratios.get(sk, {})
                if s.get("status") == "ok":
                    tput = rt.get("throughput_sps") or s.get("prod_sweeps_per_s") or 0.0
                    spd = rt.get("speedup_vs_baseline")
                    scl = rt.get("scaling_vs_smallest")
                    wall = s.get("wall_total_s") or 0.0
                    lines.append(
                        f"| `{app}` | {s['NK']} | {tput:.3f} | "
                        f"{('%.3f' % spd) if spd is not None else '—'} | "
                        f"{('%.3f' % scl) if scl is not None else '—'} | "
                        f"{wall:.2f} |"
                    )
                else:
                    lines.append(
                        f"| `{app}` | {s.get('NK', '?')} | "
                        f"FAIL ({s.get('status','?')}) | — | — | — |"
                    )
        lines.append("")
    tables_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[aggr] wrote {tables_path}", flush=True)

    # Plots.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[aggr] matplotlib unavailable: {e}", flush=True)
        return

    nk_vals = [N * K for N, K in SIZES]

    def line_data(field: str, source: str = "ratios") -> dict:
        """Return {app: [(NK, value), ...]} for ok-status entries only."""
        data = {}
        for a, r in records.items():
            sizes = r.get("sizes", {})
            ratios = r.get("ratios", {})
            pts = []
            for sk, (N, K) in zip(size_keys, SIZES):
                s = sizes.get(sk, {})
                rt = ratios.get(sk, {})
                if s.get("status") != "ok":
                    continue
                if source == "ratios":
                    v = rt.get(field)
                else:
                    v = s.get(field)
                if v is None:
                    continue
                pts.append((N * K, v))
            if pts:
                data[a] = pts
        return data

    panels = [
        ("throughput_sps",            "Throughput (prod sweeps/s)",          BENCH_DIR / "plot_throughput.png", True),
        ("speedup_vs_baseline",       f"Speedup vs {BASELINE_APP}",          BENCH_DIR / "plot_speedup.png",   True),
        ("scaling_vs_smallest",       "Scaling (per-sweep s normalised to NK=250)", BENCH_DIR / "plot_scaling.png",   True),
        ("wall_total_s",              "Wall-clock total seconds",            BENCH_DIR / "plot_wallclock.png", True),
    ]

    for field, ylabel, png, log in panels:
        data = line_data(field, source="sizes" if field == "wall_total_s" else "ratios")
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(11, 7))
        for a, pts in sorted(data.items()):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=a)
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("NK (N × K)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " — all apps (N in {25,100,1000}, K=10)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"[aggr] wrote {png}", flush=True)


LOCK_FILE = BENCH_DIR / ".iter.lock"
LOCK_STALE_S = 60 * 60 * 2   # treat a lock older than 2h as stale (orphaned)


def acquire_lock() -> bool:
    """True iff this process now holds the lock. False means another instance
    is active (or recent) and we should bow out."""
    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age < LOCK_STALE_S:
            print(f"[iter] lock present (age={age:.0f}s); another iteration is active. Exiting.",
                  flush=True)
            return False
        print(f"[iter] stale lock (age={age:.0f}s) -> overriding", flush=True)
        try: LOCK_FILE.unlink()
        except Exception: pass
    LOCK_FILE.write_text(f"pid={os.getpid()} t={time.time()}\n", encoding="utf-8")
    return True


def release_lock() -> None:
    try: LOCK_FILE.unlink()
    except Exception: pass


def bench_one_app() -> int:
    """Benchmark exactly one app (the next un-benched one). Returns 0 on success,
    1 if nothing left, 2 on lock conflict."""
    apps = discover_apps()
    n_done = sum(1 for a in apps if already_benched(a))
    print(f"[iter] discovered {len(apps)} apps; benched {n_done}/{len(apps)}", flush=True)
    if n_done >= len(apps):
        print("[iter] all apps done -> building aggregates", flush=True)
        build_aggregates()
        return 1

    app = pick_next_app(apps)
    if app is None:
        return 1
    print(f"[iter] picking: {app}", flush=True)
    t0 = time.time()
    sizes_results = bench_app(app)
    baseline = load_app_record(BASELINE_APP) if app != BASELINE_APP else None
    record = {
        "app": app,
        "started_at": t0,
        "ended_at": time.time(),
        "tags": dict(zip(("hw", "framework", "algo"), app_backend_tags(app))),
        "sizes": sizes_results,
    }
    record["ratios"] = compute_ratios(app, sizes_results, baseline)
    record["bug_check"] = bug_check(sizes_results)
    save_app_record(app, record)
    print(f"[iter] wrote benchmark_data/{app}.json", flush=True)
    n_done_after = sum(1 for a in apps if already_benched(a))
    if n_done_after >= len(apps):
        print("[iter] all 27 apps benched -> building aggregates", flush=True)
        build_aggregates()
    return 0


def main() -> int:
    drain = "--all" in sys.argv
    if not acquire_lock():
        return 2
    try:
        if drain:
            while True:
                rc = bench_one_app()
                if rc != 0:
                    return rc
                # Refresh lock mtime so other watchers see we are alive.
                LOCK_FILE.write_text(f"pid={os.getpid()} t={time.time()}\n", encoding="utf-8")
        else:
            return bench_one_app()
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
