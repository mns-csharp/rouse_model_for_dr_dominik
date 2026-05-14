"""One-shot fix for the aggregate build.

run_one() in bench_loop_iteration.py read summary.get('prod_sweeps_per_s') at
the TOP level, but per-cell summary.json nests timing under summary['timing'].
Result: every per-app JSON has throughput/per_sweep null, so only the
wall-clock plot built (wall_total_s is measured by the orchestrator itself).

This script walks each <app>.json in benchmark_data/, refills the missing
fields from _runs/<app>_N<N>_K<K>/summary.json, recomputes ratios, and
re-runs build_aggregates().
"""
from __future__ import annotations

import json
from pathlib import Path

import bench_loop_iteration as bli


def repair_one_record(app: str, rec: dict) -> dict:
    sizes = rec.get("sizes", {})
    for size_key, cell in sizes.items():
        if cell.get("status") != "ok":
            continue
        run_dir = bli.RUNS_DIR / f"{app}_{size_key}"
        sj = run_dir / "summary.json"
        if not sj.exists():
            continue
        try:
            s = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            continue
        timing = s.get("timing", {}) or {}
        prod_wall = timing.get("prod_wall_s")
        prod_sps = timing.get("prod_sweeps_per_s")
        eq_wall = timing.get("eq_wall_s")
        re2 = (s.get("observables", {}) or {}).get("re2_mean")
        rg2 = (s.get("observables", {}) or {}).get("rg2_mean")
        if prod_wall is not None:
            cell["prod_wall_s"] = prod_wall
        if prod_sps is not None:
            cell["prod_sweeps_per_s"] = prod_sps
            cell["per_sweep_s"] = (prod_wall / bli.PROD_SWEEPS) if prod_wall else 1.0 / prod_sps
        if eq_wall is not None:
            cell["eq_wall_s"] = eq_wall
        if re2 is not None and cell.get("Re2_mean") is None:
            cell["Re2_mean"] = re2
        if rg2 is not None and cell.get("Rg2_mean") is None:
            cell["Rg2_mean"] = rg2
    rec["sizes"] = sizes
    return rec


def main() -> int:
    apps = bli.discover_apps()
    baseline = None  # filled after baseline is repaired

    # First pass: repair baseline so its prod_sweeps_per_s is non-null
    bp = bli.BENCH_DIR / f"{bli.BASELINE_APP}.json"
    if bp.exists():
        b = json.loads(bp.read_text(encoding="utf-8"))
        b = repair_one_record(bli.BASELINE_APP, b)
        b["ratios"] = bli.compute_ratios(bli.BASELINE_APP, b["sizes"], None)
        bp.write_text(json.dumps(b, indent=2, default=str), encoding="utf-8")
        baseline = b
        print(f"[regen] repaired baseline: {bli.BASELINE_APP}")

    for app in apps:
        if app == bli.BASELINE_APP:
            continue
        f = bli.BENCH_DIR / f"{app}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        rec = repair_one_record(app, rec)
        rec["ratios"] = bli.compute_ratios(app, rec["sizes"], baseline)
        f.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        print(f"[regen] repaired: {app}")

    print("[regen] rebuilding aggregates ...")
    bli.build_aggregates()
    print("[regen] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
