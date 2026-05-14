"""Produce four "raw, without normalization" plots.

Companion to the existing four ratio plots written by
bench_loop_iteration.build_aggregates(). Each raw plot replaces the ratio
y-axis with the underlying raw quantity:

  plot_raw_throughput_without_normalization  : prod_sweeps_per_s
  plot_raw_speedup_without_normalization     : prod_sweeps_per_s
  plot_raw_scaling_without_normalization     : per_sweep_s
  plot_raw_wallclock_without_normalization   : wall_total_s

Each line is annotated at its rightmost ok cell with the short app code
(first two underscore-separated tokens of the directory name, e.g. c1c_nb,
g1c_ptgcl). No legend block.
"""
from __future__ import annotations

import bench_loop_iteration as bli


def short_code(app: str) -> str:
    return "_".join(app.split("_")[:2])


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apps = bli.discover_apps()
    records = {a: bli.load_app_record(a) for a in apps}
    records = {a: r for a, r in records.items() if r is not None}
    size_keys = [f"N{N}_K{K}" for N, K in bli.SIZES]

    def line_data(field: str) -> dict:
        data = {}
        for a, r in records.items():
            sizes = r.get("sizes", {})
            pts = []
            for sk, (N, K) in zip(size_keys, bli.SIZES):
                s = sizes.get(sk, {})
                if s.get("status") != "ok":
                    continue
                v = s.get(field)
                if v is None:
                    continue
                pts.append((N * K, v))
            if pts:
                data[a] = pts
        return data

    panels = [
        ("prod_sweeps_per_s", "Raw throughput (prod sweeps / s)",
         "plot_raw_throughput_without_normalization.png"),
        ("prod_sweeps_per_s", "Raw throughput (prod sweeps / s) — speedup numerator",
         "plot_raw_speedup_without_normalization.png"),
        ("per_sweep_s", "Raw per-sweep time (s)",
         "plot_raw_scaling_without_normalization.png"),
        ("wall_total_s", "Raw wall-clock total (s)",
         "plot_raw_wallclock_without_normalization.png"),
    ]

    for field, ylabel, fname in panels:
        data = line_data(field)
        if not data:
            print(f"[rawplot] {fname}: no data, skipping")
            continue
        fig, ax = plt.subplots(figsize=(20, 12))
        xmax = 0
        for a, pts in sorted(data.items()):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            line, = ax.plot(xs, ys, marker="o")
            ax.annotate(short_code(a), xy=(xs[-1], ys[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=10, va="center", ha="left",
                        color=line.get_color())
            if xs[-1] > xmax:
                xmax = xs[-1]
        ax.set_xscale("log")
        ax.set_xlim(right=xmax * 1.25)
        ax.set_xlabel("NK (N x K)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " — all apps (N in {25,100,1000}, K=10)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out = bli.BENCH_DIR / fname
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[rawplot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
