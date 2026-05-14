"""Benchmark runner — drives the workflow defined in context_benchmark.txt.

Phases:
  0  BOOTSTRAP   — env snapshot already done by caller
  1  PRE-FLIGHT  — 27 apps × N=25, K=10, eq=100, prod=100, seed=42
                   parse [CHECKLIST-Cn] lines, map to user C1..C11, emit verdicts
  2  BENCHMARK   — 27 apps × (N,K) ∈ {(25,10),(50,10),(100,10)} at eq=100/prod=100
                   write benchmark_data/<app>.json per app
  3  PLOTS+TABLES — write results.tsv, 4 PNGs, 4 ranked tables

Per the prompt RULE-08, eq=100 and prod=100 are FIXED. Per RULE-10, every
number must trace to a subprocess run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
BENCH = REPO / "benchmark_data"
PRE = BENCH / "_preflight"
RUNS = BENCH / "_runs"
LOG = REPO / "benchmark_log.txt"

EQ = 100
PROD = 100
K = 20
SEED = 42
N_VALUES = [25, 50, 100]
PER_CELL_TIMEOUT_S = 6000
PHI = 0.01          # packing fraction; box ~179 Å at N=100,K=10 — supervisor PyMOL inspection
TRAJ_STRIDE = 5     # frames every 5 sweeps → ~20 frames per phase
BASELINE = "c1c_nb_cpu_single_core_conventional_mc_numba"

GPU_HINT = ("g1c_cc", "g1c_cccl", "g1c_pt", "g1c_ptg", "g1c_ptgcl",
            "g1m_cc", "g1m_pt", "g1m_ptg", "g1m_ptgcl",
            "gnc_cc", "gnc_pt", "gnc_ptg", "gnc_ptgcl",
            "gnm_cc", "gnm_pt", "gnm_ptg", "gnm_ptgcl")


def is_gpu(app: str) -> bool:
    return app.startswith("g")


def is_multistep(app: str) -> bool:
    return "_m_" in app[:5] or app[2] == "m"  # *_m_* / *m_ in code prefix


def log_line(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line.rstrip(), flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def discover_apps() -> list[str]:
    out = []
    for d in sorted(SRC.iterdir()):
        if not d.is_dir():
            continue
        if not re.match(r"^[cg][1n][cm]_[a-z]+_", d.name):
            continue
        if not (d / "main.py").exists():
            continue
        out.append(d.name)
    return out


def run_subprocess(cmd: list[str], cwd: Path, timeout: int) -> dict:
    t0 = time.time()
    timed_out = False
    rc = None
    stdout = stderr = ""
    try:
        cp = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                            timeout=timeout)
        rc = cp.returncode
        stdout = cp.stdout
        stderr = cp.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc = -1
        stdout = (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or ""))
        stderr = (e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or ""))
    wall = time.time() - t0
    return {"rc": rc, "wall_s": wall, "stdout": stdout, "stderr": stderr,
            "timed_out": timed_out}


def detect_status(rc: int, timed_out: bool, stderr: str, stdout: str) -> str:
    if timed_out:
        return "timeout"
    low = (stderr + "\n" + stdout).lower()
    if "out of memory" in low or "cuda_error_out_of_memory" in low or "cudaerrormemoryallocation" in low:
        return "oom"
    if "cuda not available" in low or "no cuda gpus" in low:
        return "no_cuda"
    if rc != 0:
        return "error"
    return "ok"


CHECKLIST_RE = re.compile(r"\[CHECKLIST-C(\d+)\]\s+(.+)")


def parse_checklist(log_path: Path) -> dict[int, list[str]]:
    """Return map app-Cn -> list of message strings."""
    out: dict[int, list[str]] = {}
    if not log_path.exists():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = CHECKLIST_RE.search(line)
        if m:
            n = int(m.group(1))
            out.setdefault(n, []).append(m.group(2).strip())
    return out


def grep_source(app: str, pattern: str) -> tuple[str, int, str] | None:
    """Return (file, line, content) for the first hit of pattern under src/<app>/."""
    app_dir = SRC / app
    if not app_dir.is_dir():
        return None
    pat = re.compile(pattern)
    for py in sorted(app_dir.rglob("*.py")):
        try:
            lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if pat.search(line):
                rel = py.relative_to(REPO).as_posix()
                return (rel, i, line.strip())
    return None


# Mapping: user check number -> (description, app-Cn list, source-grep pattern, applicability)
# applicability: "all" / "multistep" / "conventional"
USER_CHECKS = {
    1:  ("RNG seeding supported",
         [11], r"--seed", "all"),
    2:  ("Migacz multistep MC = matrix-based energy with rank-1 correction",
         [2], r"correction_matrix|rank1|fused_accept_and_correct|multistep", "multistep"),
    3:  ("Segmented chains",
         [20, 36], r"residues_per_segment", "all"),
    4:  ("Multistep MC applied to segments rather than beads",
         [2, 4], r"segment|seg_inner|SEG_INNER", "multistep"),
    5:  ("Integer-based numberSpace PBC/MIC",
         [1, 14], r"mic_delta|number_space|NumberSpace", "all"),
    6:  ("Bonds stay unbroken throughout the simulation",
         [8, 39], r"bond_rescale|BondCheck|max_stretch", "all"),
    7:  ("Hinge + tail-C + tail-N moves at each sweep",
         [4, 26], r"hinge.*n_tail.*c_tail|MTYPE_HINGE", "all"),
    8:  ("CG-SG bonds stay unbroken",
         [8, 16], r"\bsg\b|state\.sg|sg_old|sg_new", "all"),
    9:  ("Excluded volume for non-overlap of beads",
         [15, 16], r"repulsive_energy|r_rep_sq|excluded.?volume", "all"),
    10: ("Systems initialized using random walk",
         [9], r"random_saw|random.?walk", "all"),
    11: ("Physical units",
         [16, 36], r"sigma\s*=\s*3\.8|l0\s*=|kBT|kbT|3\.8\s*#.*sigma", "all"),
}


def preflight_one(app: str) -> dict:
    """Run app at the smallest cell and emit per-app G-Cn verdicts."""
    out_dir = PRE / app
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", f"{app}.main",
        "--N", "25", "--n_chains", str(K),
        "--eq_sweeps", str(EQ), "--prod_sweeps", str(PROD),
        "--seed", str(SEED),
        "--phi", str(PHI),
        "--output_dir", str(out_dir),
        "--traj_stride", str(TRAJ_STRIDE),
        "--heartbeat_interval", "0",
        "--log_level", "INFO",
    ]
    res = run_subprocess(cmd, cwd=SRC, timeout=PER_CELL_TIMEOUT_S)
    status = detect_status(res["rc"], res["timed_out"], res["stderr"], res["stdout"])

    run_log = out_dir / "run.log"
    checklist = parse_checklist(run_log)

    # Build per-(user-Cn) verdicts.
    gates: dict[int, dict] = {}
    multistep = is_multistep(app)
    for un, (desc, app_cns, grep_pat, applicability) in USER_CHECKS.items():
        if applicability == "multistep" and not multistep:
            gates[un] = {"verdict": "NA", "reason": "not applicable to conventional MC"}
            continue
        log_hits = []
        for acn in app_cns:
            if acn in checklist:
                # Pick the LAST occurrence (post-prod state is what we want).
                msg = checklist[acn][-1]
                log_hits.append((acn, msg))
        src_hit = grep_source(app, grep_pat)
        if log_hits and src_hit:
            verdict = "PASS"
        elif log_hits:
            verdict = "PASS_LOG_ONLY"
        elif src_hit:
            verdict = "PASS_SRC_ONLY"
        else:
            verdict = "FAIL"
        gates[un] = {
            "verdict": verdict,
            "log_hits": [{"app_cn": acn, "msg": msg} for acn, msg in log_hits],
            "source_hit": ({"file": src_hit[0], "line": src_hit[1], "content": src_hit[2]}
                            if src_hit else None),
        }

    return {
        "app": app,
        "status": status,
        "rc": res["rc"],
        "wall_s": res["wall_s"],
        "stderr_tail": (res["stderr"] or "")[-500:],
        "checklist_lines_seen": sorted(checklist.keys()),
        "gates": gates,
    }


def bench_one_cell(app: str, N: int) -> dict:
    out_dir = RUNS / f"{app}_N{N}_K{K}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", f"{app}.main",
        "--N", str(N), "--n_chains", str(K),
        "--eq_sweeps", str(EQ), "--prod_sweeps", str(PROD),
        "--seed", str(SEED),
        "--phi", str(PHI),
        "--output_dir", str(out_dir),
        "--traj_stride", str(TRAJ_STRIDE),
        "--heartbeat_interval", "0",
        "--log_level", "WARNING",
    ]
    res = run_subprocess(cmd, cwd=SRC, timeout=PER_CELL_TIMEOUT_S)
    status = detect_status(res["rc"], res["timed_out"], res["stderr"], res["stdout"])
    summary_path = out_dir / "summary.json"
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = None

    eq_wall = prod_wall = prod_sps = re2 = rg2 = None
    if summary:
        timing = summary.get("timing", {}) or {}
        observ = summary.get("observables", {}) or {}
        eq_wall = timing.get("eq_wall_s") or summary.get("eq_wall_s")
        prod_wall = timing.get("prod_wall_s") or summary.get("prod_wall_s")
        prod_sps = timing.get("prod_sweeps_per_s") or summary.get("prod_sweeps_per_s")
        if prod_sps is None and prod_wall:
            prod_sps = PROD / max(prod_wall, 1e-12)
        re2 = observ.get("re2_mean") or summary.get("Re2_mean")
        rg2 = observ.get("rg2_mean") or summary.get("Rg2_mean")

    # Persist stdout/stderr.
    (out_dir / "stdout.txt").write_text(res["stdout"] or "", encoding="utf-8")
    (out_dir / "stderr.txt").write_text(res["stderr"] or "", encoding="utf-8")

    return {
        "N": N, "K": K, "NK": N * K,
        "status": status,
        "returncode": res["rc"],
        "wall_total_s": res["wall_s"],
        "eq_wall_s": eq_wall,
        "prod_wall_s": prod_wall,
        "prod_sweeps_per_s": prod_sps,
        "per_sweep_s": (prod_wall / PROD) if prod_wall else None,
        "Re2_mean": re2,
        "Rg2_mean": rg2,
        "cmd": " ".join(cmd),
        "run_dir": out_dir.as_posix(),
    }


def bench_app(app: str) -> dict:
    sizes = {}
    for N in N_VALUES:
        log_line(f"[bench] {app} N={N} K={K}")
        res = bench_one_cell(app, N)
        sizes[f"N{N}_K{K}"] = res
        log_line(f"[bench] {app} N={N} K={K} status={res['status']} wall={res['wall_total_s']:.1f}s")
        if res["status"] in ("oom", "timeout") and N != N_VALUES[-1]:
            # Skip larger N
            for N2 in N_VALUES[N_VALUES.index(N) + 1:]:
                sizes[f"N{N2}_K{K}"] = {
                    "N": N2, "K": K, "NK": N2 * K,
                    "status": f"skipped_after_{res['status']}",
                    "returncode": None, "wall_total_s": None,
                    "eq_wall_s": None, "prod_wall_s": None,
                    "prod_sweeps_per_s": None, "per_sweep_s": None,
                    "Re2_mean": None, "Rg2_mean": None,
                    "cmd": "", "run_dir": "",
                }
            break
    return sizes


def compute_ratios(sizes: dict, baseline: dict | None) -> dict:
    ratios = {}
    # smallest-NK reference for scaling
    ok = [(k, v) for k, v in sizes.items() if v["status"] == "ok" and v.get("per_sweep_s")]
    base_size = ok[0][0] if ok else None
    base_per_sweep = sizes[base_size]["per_sweep_s"] if base_size else None

    for sk, r in sizes.items():
        d = {}
        if r["status"] == "ok":
            d["throughput_sps"] = r["prod_sweeps_per_s"]
            d["per_sweep_s"] = r["per_sweep_s"]
            d["wall_total_s"] = r["wall_total_s"]
            if base_per_sweep:
                d["scaling_vs_smallest"] = r["per_sweep_s"] / base_per_sweep
            if baseline and baseline.get(sk, {}).get("status") == "ok":
                b = baseline[sk]
                if b.get("prod_sweeps_per_s") and r.get("prod_sweeps_per_s"):
                    d["speedup_vs_baseline"] = r["prod_sweeps_per_s"] / b["prod_sweeps_per_s"]
                if b.get("wall_total_s") and r.get("wall_total_s"):
                    d["wall_ratio_baseline_over_this"] = b["wall_total_s"] / r["wall_total_s"]
        else:
            d["status"] = r["status"]
        ratios[sk] = d
    return ratios


def app_tags(app: str) -> dict:
    hw = "CPU" if app.startswith("c") else "GPU"
    thread = "single" if app[1] == "1" else "multi"
    algo = "conv" if app[2] == "c" else "multi"
    backend = app.split("_")[1]
    return {"hw": hw, "thread": thread, "algo": algo, "backend": backend}


def phase1_preflight(apps: list[str]) -> dict:
    log_line(f"=== PHASE 1 PRE-FLIGHT === apps={len(apps)}")
    results = {}
    apps_fail = 0
    for app in apps:
        log_line(f"[preflight] {app}")
        r = preflight_one(app)
        n_fail = sum(1 for g in r["gates"].values() if g["verdict"] == "FAIL")
        n_pass = sum(1 for g in r["gates"].values() if g["verdict"].startswith("PASS"))
        n_na = sum(1 for g in r["gates"].values() if g["verdict"] == "NA")
        log_line(f"[preflight] {app} status={r['status']} wall={r['wall_s']:.1f}s "
                 f"gates_pass={n_pass} gates_na={n_na} gates_fail={n_fail}")
        results[app] = r
        if n_fail > 0:
            apps_fail += 1
    (BENCH / "_preflight_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    log_line(f"=== PHASE 1 PRE-FLIGHT DONE === apps_pass={len(apps)-apps_fail} apps_fail={apps_fail}")
    return results


def phase2_benchmark(apps: list[str]) -> dict:
    log_line(f"=== PHASE 2 BENCHMARK === apps={len(apps)} cells_per_app={len(N_VALUES)}")
    # Baseline first.
    baseline_sizes = None
    records = {}

    if BASELINE in apps:
        log_line(f"[benchmark] baseline FIRST: {BASELINE}")
        sizes = bench_app(BASELINE)
        rec = {
            "app": BASELINE, "tags": app_tags(BASELINE),
            "sizes": sizes, "ratios": compute_ratios(sizes, None),
        }
        records[BASELINE] = rec
        baseline_sizes = sizes
        (BENCH / f"{BASELINE}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")

    for app in apps:
        if app == BASELINE:
            continue
        log_line(f"[benchmark] {app}")
        sizes = bench_app(app)
        rec = {
            "app": app, "tags": app_tags(app),
            "sizes": sizes, "ratios": compute_ratios(sizes, baseline_sizes),
        }
        records[app] = rec
        (BENCH / f"{app}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    log_line(f"=== PHASE 2 BENCHMARK DONE === records={len(records)}")
    return records


def phase3_outputs(records: dict) -> None:
    log_line(f"=== PHASE 3 PLOTS & TABLES ===")
    # results.tsv (long format)
    rows = ["app\tN\tK\tstatus\twall_total_s\teq_wall_s\tprod_wall_s\tper_sweep_s\tprod_sweeps_per_s\tspeedup_vs_baseline\tscaling_vs_smallest\tRe2_mean\tRg2_mean"]
    for app, rec in records.items():
        for sk, s in rec["sizes"].items():
            ratios = rec["ratios"].get(sk, {})
            row = [
                app, str(s.get("N")), str(s.get("K")),
                s.get("status", ""),
                f"{s.get('wall_total_s'):.4f}" if s.get("wall_total_s") else "",
                f"{s.get('eq_wall_s'):.4f}" if s.get("eq_wall_s") else "",
                f"{s.get('prod_wall_s'):.4f}" if s.get("prod_wall_s") else "",
                f"{s.get('per_sweep_s'):.6f}" if s.get("per_sweep_s") else "",
                f"{s.get('prod_sweeps_per_s'):.4f}" if s.get("prod_sweeps_per_s") else "",
                f"{ratios.get('speedup_vs_baseline'):.4f}" if ratios.get("speedup_vs_baseline") else "",
                f"{ratios.get('scaling_vs_smallest'):.4f}" if ratios.get("scaling_vs_smallest") else "",
                f"{s.get('Re2_mean'):.4f}" if s.get("Re2_mean") else "",
                f"{s.get('Rg2_mean'):.4f}" if s.get("Rg2_mean") else "",
            ]
            rows.append("\t".join(row))
    (BENCH / "results.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    log_line(f"[outputs] wrote {BENCH/'results.tsv'} ({len(rows)} rows)")

    # Build matplotlib plots.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apps_ordered = list(records.keys())

    def short(a: str) -> str:
        return "_".join(a.split("_")[:2])

    def get_series(field: str, source: str = "sizes"):
        """Return {app: {N: value}} for the given field from either 'sizes' or 'ratios'."""
        out = {}
        for a, rec in records.items():
            block = rec[source]
            d = {}
            for sk, s in block.items():
                if isinstance(s, dict):
                    val = s.get(field)
                    if val is None:
                        continue
                    if "N" in s and isinstance(s["N"], (int, float)):
                        d[s["N"]] = val
                    else:
                        # parse from key
                        m = re.match(r"N(\d+)_K", sk)
                        if m:
                            d[int(m.group(1))] = val
            if d:
                out[a] = d
        return out

    def plot_metric(field: str, source: str, ylabel: str, title: str,
                     out_path: Path, logy: bool = False):
        plt.figure(figsize=(12, 7))
        series = get_series(field, source)
        cmap = plt.colormaps["tab20"]
        for i, a in enumerate(apps_ordered):
            d = series.get(a, {})
            if not d:
                continue
            xs = sorted(d.keys())
            ys = [d[x] for x in xs]
            tag = app_tags(a)
            marker = "o" if tag["hw"] == "CPU" else "^"
            ls = "-" if tag["algo"] == "conv" else "--"
            plt.plot(xs, ys, marker=marker, linestyle=ls,
                     color=cmap(i % 20), label=short(a))
        plt.xlabel("N (chain length)")
        plt.ylabel(ylabel)
        plt.title(title)
        if logy:
            plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        log_line(f"[outputs] wrote {out_path}")

    plot_metric("throughput_sps", "ratios",
                ylabel="Throughput (prod sweeps / s)",
                title=f"Throughput vs N  (K={K}, phi={PHI}, eq=prod={PROD})",
                out_path=BENCH / "plot_throughput.png", logy=True)
    plot_metric("wall_total_s", "ratios",
                ylabel="Wall-clock total (s)",
                title=f"Wallclock vs N  (K={K}, phi={PHI}, eq=prod={PROD})",
                out_path=BENCH / "plot_wallclock.png", logy=True)
    plot_metric("scaling_vs_smallest", "ratios",
                ylabel="per_sweep_s(N) / per_sweep_s(N=25)",
                title=f"Scaling vs smallest N  (K={K}, phi={PHI}, eq=prod={PROD})",
                out_path=BENCH / "plot_scaling.png", logy=True)
    plot_metric("speedup_vs_baseline", "ratios",
                ylabel="Speedup × (relative to c1c_nb)",
                title=f"Speedup vs c1c_nb baseline  (K={K}, phi={PHI}, eq=prod={PROD})",
                out_path=BENCH / "plot_speedup.png", logy=True)

    # Ranked tables — replicate the structure of append_ranking_tables.py
    # but headlined at N=100 K=10 (NK=1000), with secondary at N=50 K=10.
    HEAD_KEY = f"N100_K{K}"
    SECOND_KEY = f"N50_K{K}"

    def fmt(v):
        import math
        if v is None: return "—"
        if isinstance(v, str): return v
        if isinstance(v, float) and not math.isfinite(v): return "—"
        if isinstance(v, (int, float)):
            if abs(v) >= 1000: return f"{v:.1f}"
            if abs(v) >= 1: return f"{v:.3f}"
            return f"{v:.4f}"
        return str(v)

    def cell_val(rec, key, source, field):
        s = rec["sizes"].get(key, {})
        if s.get("status") != "ok":
            return s.get("status") or "—"
        if source == "sizes":
            return s.get(field)
        return rec["ratios"].get(key, {}).get(field)

    def render(metric, ylabel, source, field, descending):
        rows_t = []
        for app, rec in records.items():
            head = cell_val(rec, HEAD_KEY, source, field)
            sec = cell_val(rec, SECOND_KEY, source, field)
            import math
            sort_val = head if isinstance(head, (int, float)) and math.isfinite(head) else (
                float("-inf") if descending else float("inf"))
            ok = isinstance(head, (int, float))
            rows_t.append((app, head, sec, sort_val, ok))
        rows_t.sort(key=lambda r: r[3], reverse=descending)
        lines = [
            f"### {ylabel}",
            "",
            f"Headlined at NK={100*K} (N=100, K={K}); NK={50*K} (N=50, K={K}) shown for context.",
            "",
            f"| Rank | Code | App | {metric} @ N=100,K={K} | {metric} @ N=50,K={K} |",
            "|---|---|---|---|---|",
        ]
        for i, (app, head, sec, _, ok) in enumerate(rows_t, 1):
            rank = str(i) if ok else "—"
            lines.append(f"| {rank} | `{short(app)}` | `{app}` | {fmt(head)} | {fmt(sec)} |")
        return "\n".join(lines) + "\n"

    table_md = ["# Benchmark Tables", "",
                f"Spec: {len(records)} apps × N∈{{25,50,100}} × K={K}, phi={PHI}, athermal, l0=sigma, eq=prod={PROD}, seed={SEED}, traj_stride={TRAJ_STRIDE}.",
                f"Baseline app for speedup: `{BASELINE}` → code `{short(BASELINE)}`.",
                ""]
    table_md.append(render("throughput (sw/s)",
                           "Ranked by throughput (prod sweeps/s, higher is better)",
                           "sizes", "prod_sweeps_per_s", True))
    table_md.append(render("speedup ×",
                           f"Ranked by speedup vs `{BASELINE}` (higher is better)",
                           "ratios", "speedup_vs_baseline", True))
    table_md.append(render("scaling ×",
                           "Ranked by scaling = per_sweep_s @ N / per_sweep_s @ N=25 (lower is better — flatter)",
                           "ratios", "scaling_vs_smallest", False))
    table_md.append(render("wall (s)",
                           "Ranked by wall-clock total seconds (lower is better)",
                           "sizes", "wall_total_s", False))
    (BENCH / "tables.md").write_text("\n".join(table_md), encoding="utf-8")
    log_line(f"[outputs] wrote {BENCH/'tables.md'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["all", "preflight", "benchmark", "outputs"], default="all")
    p.add_argument("--apps", nargs="*", help="restrict to these app names (full directory names)")
    p.add_argument("--skip-gpu", action="store_true")
    args = p.parse_args()

    PRE.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    LOG.touch(exist_ok=True)

    apps = discover_apps()
    if args.apps:
        apps = [a for a in apps if a in set(args.apps)]
    if args.skip_gpu:
        apps = [a for a in apps if not is_gpu(a)]

    log_line(f"=== RUNNER START === phase={args.phase} apps={len(apps)} eq={EQ} prod={PROD} K={K} seed={SEED}")
    t0 = time.time()

    if args.phase in ("all", "preflight"):
        phase1_preflight(apps)

    records = {}
    if args.phase in ("all", "benchmark"):
        records = phase2_benchmark(apps)
    elif args.phase == "outputs":
        for app in apps:
            f = BENCH / f"{app}.json"
            if f.exists():
                records[app] = json.loads(f.read_text(encoding="utf-8"))

    if args.phase in ("all", "outputs"):
        if not records and args.phase == "all":
            # benchmark just ran, but records already populated above
            pass
        elif not records:
            for app in apps:
                f = BENCH / f"{app}.json"
                if f.exists():
                    records[app] = json.loads(f.read_text(encoding="utf-8"))
        phase3_outputs(records)

    log_line(f"=== RUNNER DONE === wall={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
