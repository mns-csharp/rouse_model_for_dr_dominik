"""bench_select — rank the 16 apps by throughput and identify winner X.

Reads <verif_root>/observables_summary.tsv (produced by verify_runner.py)
and computes per-app throughput. Ranks descending; emits TSV + bar plot.

Headline metric: per_sweep_s at N=100, phi=0.10 (the heaviest cell —
most differentiating). Falls back to median over all cells if N=100/phi=0.10
is missing.

Usage:
    python src/bench_select.py --verif_root _verif_full
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROD_SWEEPS = 500   # what verify_runner.py used


def parse_obs(path: Path) -> List[Dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                rows.append({
                    "app": r["app"],
                    "N": int(r["N"]),
                    "phi": float(r["phi"]),
                    "seed": int(r["seed"]),
                    "wall_eq": float(r["wall_eq"]),
                    "wall_prod": float(r["wall_prod"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def throughput_per_app(rows: List[Dict]) -> Dict[str, Dict]:
    """For each app: per_sweep_s_headline (N=100, phi=0.10), per_sweep_s_median,
    n_cells_seen."""
    by_app: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_app[r["app"]].append(r)
    out: Dict[str, Dict] = {}
    for app, rs in by_app.items():
        # Headline: N=100, phi=0.10 (any seed; use median across seeds).
        head = [r for r in rs if r["N"] == 100 and abs(r["phi"] - 0.10) < 1e-6]
        head_per_sweep = []
        for r in head:
            if r["wall_prod"] > 0 and math.isfinite(r["wall_prod"]):
                head_per_sweep.append(PROD_SWEEPS / r["wall_prod"])
        head_med = (statistics.median(head_per_sweep)
                    if head_per_sweep else float("nan"))

        # Median across all cells (fallback).
        all_per_sweep = []
        for r in rs:
            if r["wall_prod"] > 0 and math.isfinite(r["wall_prod"]):
                all_per_sweep.append(PROD_SWEEPS / r["wall_prod"])
        all_med = (statistics.median(all_per_sweep)
                   if all_per_sweep else float("nan"))

        out[app] = {
            "throughput_at_N100_phi0.10": head_med,
            "throughput_median_all_cells": all_med,
            "n_cells_seen": len(rs),
            "n_cells_at_headline": len(head),
        }
    return out


def write_ranking(per_app: Dict[str, Dict], path: Path) -> None:
    rows = []
    for app, d in per_app.items():
        rows.append({
            "app": app,
            "throughput_at_N100_phi0.10": d["throughput_at_N100_phi0.10"],
            "throughput_median_all_cells": d["throughput_median_all_cells"],
            "n_cells_seen": d["n_cells_seen"],
            "n_cells_at_headline": d["n_cells_at_headline"],
        })

    def sort_key(r):
        v = r["throughput_at_N100_phi0.10"]
        if not math.isfinite(v):
            v = r["throughput_median_all_cells"]
        return -v if math.isfinite(v) else float("inf")

    rows.sort(key=sort_key)
    fieldnames = ["app", "throughput_at_N100_phi0.10",
                  "throughput_median_all_cells",
                  "n_cells_seen", "n_cells_at_headline"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader(); w.writerows(rows)
    return rows


def render_plot(rows: List[Dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[bench_select] matplotlib import failed: {e}")
        return
    apps = [r["app"] for r in rows]
    vals = [r["throughput_at_N100_phi0.10"]
             if math.isfinite(r["throughput_at_N100_phi0.10"])
             else r["throughput_median_all_cells"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 8))
    y = list(range(len(apps)))
    ax.barh(y, vals, color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels(apps, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("prod_sweeps_per_s (median across seeds, at N=100 phi=0.10)")
    ax.set_title(f"Throughput ranking — {len(apps)} apps")
    for yi, v in zip(y, vals):
        if math.isfinite(v):
            ax.text(v, yi, f"  {v:.1f}", va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verif_root", default="_verif_full")
    args = ap.parse_args()

    root = (Path(__file__).resolve().parent.parent / args.verif_root).resolve()
    obs_path = root / "observables_summary.tsv"
    if not obs_path.exists():
        print(f"[bench_select] missing {obs_path}", file=sys.stderr)
        return 1

    rows = parse_obs(obs_path)
    if not rows:
        print(f"[bench_select] {obs_path} is empty", file=sys.stderr)
        return 1

    per_app = throughput_per_app(rows)
    ranking = write_ranking(per_app, root / "bench_ranking.tsv")
    render_plot(ranking, root / "bench_ranking_plot.png")

    print(f"[bench_select] {len(ranking)} apps ranked (data: {len(rows)} cells from {obs_path.name})")
    print()
    print(f"{'rank':<5}{'app':<48}{'tput@N100,phi0.10':>20}{'tput_med_all':>15}")
    for i, r in enumerate(ranking, 1):
        head = r["throughput_at_N100_phi0.10"]
        med = r["throughput_median_all_cells"]
        head_s = f"{head:.2f}" if math.isfinite(head) else "n/a"
        med_s = f"{med:.2f}" if math.isfinite(med) else "n/a"
        print(f"{i:<5}{r['app']:<48}{head_s:>20}{med_s:>15}")

    print()
    if ranking and math.isfinite(ranking[0]["throughput_at_N100_phi0.10"]):
        print(f"Winner X = {ranking[0]['app']}  "
              f"(per_sweep_s = {ranking[0]['throughput_at_N100_phi0.10']:.2f})")
    elif ranking:
        print(f"Winner X = {ranking[0]['app']}  "
              f"(per_sweep_s_median = {ranking[0]['throughput_median_all_cells']:.2f})  "
              f"[no headline cell N=100/phi=0.10 available; using median]")
    else:
        print("No data.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
