"""BenchAnalyzer — reads bench_timings.tsv, emits B4-B7 log lines,
writes speedup_summary.tsv, generates scaling and heatmap plots.
"""

import csv
import logging
import math
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rouse_model_python.src.libs.analysis.slopes_computer import SlopesComputer
from rouse_model_python.src.libs.analysis.speedup_computer import SpeedupComputer
from rouse_model_python.src.libs.io.bench_logging import BenchLogging
from rouse_model_python.src.libs.io.tsv_writer import TSVWriter


class BenchAnalyzer:
    ALL_N = ("25", "50", "100")
    _logger = logging.getLogger("rouse_model_python.src.libs.analysis.bench_analyzer")

    @staticmethod
    def _read_tsv(path: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        if not os.path.exists(path):
            return rows
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                rows.append(row)
        return rows

    @staticmethod
    def _group_latest(rows):
        out = {}
        for row in rows:
            key = (row["N"], row["algorithm"], row["device"])
            out[key] = row
        return out

    @staticmethod
    def _t(key, groups):
        row = groups.get(key)
        if row is None:
            return float("nan")
        try:
            return float(row["total_wall_s"])
        except (ValueError, KeyError):
            return float("nan")

    @classmethod
    def write_speedup_summary(cls, bench_dir, groups, slopes, speedups):
        path = os.path.join(bench_dir, "speedup_summary.tsv")
        header = [
            "N", "device", "speedup_migacz",
            "speedup_gpu_conventional", "speedup_gpu_multistep",
            "slope_multistep_cpu", "slope_multistep_gpu",
            "slope_conventional_cpu", "slope_conventional_gpu",
        ]
        if os.path.exists(path):
            os.remove(path)
        for N in cls.ALL_N:
            for dev in ("cpu", "gpu"):
                row = [
                    N, dev,
                    speedups["migacz"].get((N, dev), float("nan")),
                    speedups["gpu"].get((N, "conventional"), float("nan")),
                    speedups["gpu"].get((N, "multistep"), float("nan")),
                    slopes.get(("multistep", "cpu"), float("nan")),
                    slopes.get(("multistep", "gpu"), float("nan")),
                    slopes.get(("conventional", "cpu"), float("nan")),
                    slopes.get(("conventional", "gpu"), float("nan")),
                ]
                TSVWriter.append_row(path, header, row)
        return path

    @classmethod
    def _plot_scaling(cls, bench_dir, groups, algorithm, outfile):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        for dev, marker, color in (("cpu", "o", "tab:blue"), ("gpu", "s", "tab:orange")):
            xs, ys = [], []
            for N in cls.ALL_N:
                t = cls._t((N, algorithm, dev), groups)
                if math.isfinite(t):
                    xs.append(int(N))
                    ys.append(t)
            if len(xs) >= 1:
                ax.plot(xs, ys, marker=marker, color=color, label=dev,
                        linewidth=1.5, markersize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("chain length N"); ax.set_ylabel("total wall-time (s)")
        ax.set_title(f"Wall-time scaling — {algorithm}")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(title="device")
        fig.tight_layout()
        fig.savefig(os.path.join(bench_dir, outfile), dpi=120)
        plt.close(fig)

    @classmethod
    def _plot_speedup_heatmap(cls, bench_dir, speedups):
        Ns = cls.ALL_N; devs = ("cpu", "gpu")
        Z = np.full((len(Ns), len(devs)), np.nan)
        for i, N in enumerate(Ns):
            for j, dev in enumerate(devs):
                Z[i, j] = speedups["migacz"].get((N, dev), float("nan"))
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        vmax = max(3.0, float(np.nanmax(Z)) if np.isfinite(np.nanmax(Z)) else 3.0)
        im = ax.imshow(Z, cmap="RdYlGn", vmin=0.5, vmax=vmax)
        ax.set_xticks(range(len(devs))); ax.set_xticklabels(devs)
        ax.set_yticks(range(len(Ns))); ax.set_yticklabels([f"N={n}" for n in Ns])
        ax.set_xlabel("device")
        ax.set_title("Migacz multistep speedup  (t_conv / t_multi)")
        for i in range(len(Ns)):
            for j in range(len(devs)):
                v = Z[i, j]
                label = f"{v:.2f}×" if np.isfinite(v) else "n/a"
                ax.text(j, i, label, ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, label="speedup")
        fig.tight_layout()
        fig.savefig(os.path.join(bench_dir, "bench_speedup_heatmap.png"), dpi=120)
        plt.close(fig)

    @classmethod
    def emit_gate_lines(cls, groups, slopes, speedups):
        for alg in ("multistep", "conventional"):
            tag = "B4" if alg == "multistep" else "B5"
            for dev in ("cpu", "gpu"):
                t25 = cls._t(("25", alg, dev), groups)
                t50 = cls._t(("50", alg, dev), groups)
                t100 = cls._t(("100", alg, dev), groups)
                slope = slopes.get((alg, dev), float("nan"))
                cls._logger.info("[BENCH-%s] Algorithm=%s Device=%s T25=%.6f T50=%.6f T100=%.6f Slope=%.4f",
                            tag, alg, dev, t25, t50, t100, slope)
        for N in cls.ALL_N:
            for dev in ("cpu", "gpu"):
                tc = cls._t((N, "conventional", dev), groups)
                tm = cls._t((N, "multistep", dev), groups)
                ratio = speedups["migacz"].get((N, dev), float("nan"))
                cls._logger.info("[BENCH-B6] N=%s Device=%s T_Conventional=%.6f T_Multistep=%.6f SpeedupMigacz=%.4f",
                            N, dev, tc, tm, ratio)
        for N in cls.ALL_N:
            for alg in ("multistep", "conventional"):
                tcpu = cls._t((N, alg, "cpu"), groups)
                tgpu = cls._t((N, alg, "gpu"), groups)
                ratio = speedups["gpu"].get((N, alg), float("nan"))
                cls._logger.info("[BENCH-B7] N=%s Algorithm=%s T_Cpu=%.6f T_Gpu=%.6f SpeedupGPU=%.4f",
                            N, alg, tcpu, tgpu, ratio)

    @classmethod
    def run(cls, bench_dir: str) -> dict:
        BenchLogging.configure(bench_dir)
        rows = cls._read_tsv(os.path.join(bench_dir, "bench_timings.tsv"))
        groups = cls._group_latest(rows)
        cls._logger.info("[BENCH-ANALYSIS] loaded %d bench_timings rows covering %d cells",
                    len(rows), len(groups))
        slopes = SlopesComputer.compute(groups)
        speedups = SpeedupComputer.compute(groups)
        summary = cls.write_speedup_summary(bench_dir, groups, slopes, speedups)
        cls._logger.info("[BENCH-ANALYSIS] wrote %s", summary)
        cls._plot_scaling(bench_dir, groups, "multistep", "bench_scaling_multistep.png")
        cls._plot_scaling(bench_dir, groups, "conventional", "bench_scaling_conventional.png")
        cls._plot_speedup_heatmap(bench_dir, speedups)
        cls.emit_gate_lines(groups, slopes, speedups)
        return {"slopes": slopes, "speedups": speedups}
