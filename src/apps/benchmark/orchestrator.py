"""BenchmarkOrchestrator — multi-cell driver for the Rouse benchmark protocol.

Responsibilities:
  - R7 cleanup of bench_output/06_benchmarks/ artifacts from prior runs.
  - Drive PE1, PE2, PE3 per-N (three sub-sweeps each, one per N in {25, 50, 100}).
  - Apply each PE's winner rule, emit [BENCH-PE{1,2,3}-WINNER] lines, and
    upsert the per-N row in pe_winners.tsv (the contract consumed by B-cells).
  - Drive the 12-cell B-matrix (N x algorithm x device) with --repeats 3,
    consuming pe_winners.tsv for (residues_per_segment, max_angle_hinge,
    batch_size) per R6.
  - Drive B1 smoke run and B2 dispatch verification.
  - Invoke BenchAnalyzer to compute B4-B7 gate lines and generate plots.
"""

from __future__ import annotations

import csv
import glob
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

from rouse_model_python.src.apps.benchmark.benchmark_app import BenchmarkApp
from rouse_model_python.src.libs.config.config_loader import ConfigLoader
from rouse_model_python.src.libs.io.bench_logging import BenchLogging
from rouse_model_python.src.libs.io.tsv_writer import TSVWriter


_LOGGER = logging.getLogger("rouse_model_python.src.apps.benchmark.orchestrator")


class BenchmarkOrchestrator:
    N_VALUES: Tuple[int, ...] = (25, 50, 100)
    ALGORITHMS: Tuple[str, ...] = ("conventional", "multistep")
    DEVICES: Tuple[str, ...] = ("cpu", "gpu")
    ACCEPT_LOW: float = 0.2
    ACCEPT_HIGH: float = 0.6
    GPU_UTIL_FLOOR: float = 0.40

    def __init__(self, output_dir: str = "bench_output", phi: float = 0.035,
                 n_chains_bench: int = 50, warmup_sweeps: int = 10,
                 repeats: int = 3, seed: int = 42,
                 skip_pe: bool = False, skip_b_matrix: bool = False,
                 skip_cpu_pe: bool = False,
                 pe_eq_override: Optional[int] = None,
                 pe_prod_override: Optional[int] = None,
                 pe3_eq_override: Optional[int] = None,
                 pe3_prod_override: Optional[int] = None,
                 b_timing_mode: bool = False,
                 n_replicas: int = 1):
        self.output_dir = output_dir
        self.bench_dir = os.path.join(output_dir, "06_benchmarks")
        self.phi = phi
        self.n_chains_bench = n_chains_bench
        self.warmup_sweeps = warmup_sweeps
        self.repeats = repeats
        self.seed = seed
        self.skip_pe = skip_pe
        self.skip_b_matrix = skip_b_matrix
        self.skip_cpu_pe = skip_cpu_pe
        self.pe_eq_override = pe_eq_override
        self.pe_prod_override = pe_prod_override
        self.pe3_eq_override = pe3_eq_override
        self.pe3_prod_override = pe3_prod_override
        self.b_timing_mode = b_timing_mode
        self.n_replicas = n_replicas
        self.configs = ConfigLoader.load()
        self._cuda_device_name: Optional[str] = None

    def _ensure_logging(self) -> None:
        os.makedirs(self.bench_dir, exist_ok=True)
        BenchLogging.configure(self.bench_dir)

    def run_r7_cleanup(self) -> None:
        """R7: delete previous benchmark artifacts before the first experiment."""
        os.makedirs(self.bench_dir, exist_ok=True)
        patterns = ("*.log", "*.png", "*.tsv", "*.json")
        n_deleted = 0
        for pat in patterns:
            for p in glob.glob(os.path.join(self.bench_dir, pat)):
                try:
                    os.remove(p)
                    n_deleted += 1
                except OSError:
                    pass
        # Configure logging AFTER deleting benchmark.log so the new file gets the
        # CLEANUP record as its first line.
        self._ensure_logging()
        if n_deleted == 0:
            _LOGGER.info("[CLEANUP] NoPreviousArtifacts=true directory=%s", self.bench_dir)
        else:
            _LOGGER.info("[CLEANUP] Deleted=%d files from %s", n_deleted, self.bench_dir)

    # ------------------------------------------------------------------
    # Programmatic BenchmarkApp invocation
    # ------------------------------------------------------------------
    def _invoke_cell(self, argv: List[str]) -> int:
        return BenchmarkApp(argv=argv).run()

    def _base_argv(self, *, device: str, algorithm: str, N: int,
                   n_chains: int, eq: Optional[int], prod: Optional[int],
                   cell_tag: str, pe_mode: str = "none",
                   residues_per_segment: Optional[int] = None,
                   max_angle_hinge: Optional[float] = None,
                   batch_size: Optional[int] = None,
                   warmup_sweeps: Optional[int] = None,
                   repeats: int = 1,
                   emit_b2: bool = False,
                   b1_smoke: bool = False,
                   skip_timings_row: bool = False,
                   timing_mode: bool = False,
                   n_replicas: int = 1) -> List[str]:
        argv = [
            "--device", device,
            "--is_parallel", "true" if device == "gpu" else "false",
            "--force",
            "--algorithm", algorithm,
            "--output_dir", self.output_dir,
            "--N", str(N),
            "--n_chains", str(n_chains),
            "--phi", f"{self.phi}",
            "--warmup_sweeps", str(warmup_sweeps if warmup_sweeps is not None else self.warmup_sweeps),
            "--repeats", str(repeats),
            "--cell_tag", cell_tag,
            "--pe_mode", pe_mode,
            "--seed", str(self.seed),
        ]
        if eq is not None:
            argv += ["--eq_sweeps", str(eq)]
        if prod is not None:
            argv += ["--prod_sweeps", str(prod)]
        if residues_per_segment is not None:
            argv += ["--residues_per_segment", str(residues_per_segment)]
        if max_angle_hinge is not None:
            argv += ["--max_angle_hinge", f"{max_angle_hinge:.6f}"]
        if batch_size is not None:
            argv += ["--batch_size", str(batch_size)]
        if emit_b2:
            argv += ["--emit_b2"]
        if b1_smoke:
            argv += ["--b1_smoke"]
        if skip_timings_row:
            argv += ["--skip_timings_row"]
        if timing_mode:
            argv += ["--timing_mode"]
        if n_replicas > 1:
            argv += ["--n_replicas", str(n_replicas)]
        return argv

    # ------------------------------------------------------------------
    # PE TSV readers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_pe_tsv(path: str, N_filter: int) -> List[Dict[str, str]]:
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                try:
                    if int(row["N"]) == int(N_filter):
                        rows.append(row)
                except (KeyError, ValueError):
                    continue
        return rows

    # ------------------------------------------------------------------
    # PE1 — segment-size sweep per N
    # ------------------------------------------------------------------
    def run_pe1_for_N(self, N: int) -> Dict[str, object]:
        pe1_cfg = self.configs.benchmark.pe1
        n_chains = int(pe1_cfg.get("n_chains", 20))
        eq = int(self.pe_eq_override if self.pe_eq_override is not None
                 else pe1_cfg.get("eq_sweeps", 500))
        prod = int(self.pe_prod_override if self.pe_prod_override is not None
                   else pe1_cfg.get("prod_sweeps", 500))
        candidates: List[int] = list(pe1_cfg.get("candidates", [4, 8, 16, 20, 32, 64]))
        device = "cpu" if self.skip_cpu_pe else "gpu"

        _LOGGER.info("[BENCH-PE1-BEGIN] N=%d candidates=%s device=%s",
                     N, candidates, device)
        for seg in candidates:
            cell_tag = f"PE1-N{N}-seg{seg}"
            argv = self._base_argv(
                device=device, algorithm="multistep", N=N, n_chains=n_chains,
                eq=eq, prod=prod, cell_tag=cell_tag, pe_mode="pe1",
                residues_per_segment=seg, repeats=1, skip_timings_row=True,
            )
            t0 = time.perf_counter()
            try:
                self._invoke_cell(argv)
            except Exception as exc:
                _LOGGER.error("[BENCH-PE1-ERROR] N=%d seg=%d err=%r", N, seg, exc)
            _LOGGER.info("[BENCH-PE1-CAND-DONE] N=%d seg=%d wall=%.2fs",
                         N, seg, time.perf_counter() - t0)

        tsv_path = os.path.join(self.bench_dir, "pe1_segment_sweep.tsv")
        rows = self._read_pe_tsv(tsv_path, N)
        winner = self._pick_pe1_winner(rows)
        _LOGGER.info(
            "[BENCH-PE1-WINNER] N=%d RESIDUES_PER_SEGMENT=%d PerSweepMs=%.4f "
            "HingeAccept=%.4f TailAccept=%.4f GpuUtilMean=%.4f BandMiss=%s Source=PE1",
            N, int(winner["segment_size"]), float(winner["per_sweep_ms"]),
            float(winner["hinge_accept"]), float(winner["tail_accept"]),
            float(winner["gpu_util_mean"]), winner["band_miss"],
        )
        if winner["band_miss"]:
            _LOGGER.warning(
                "[BENCH-PE1-BAND-MISS] N=%d seg=%d HingeAccept=%.4f TailAccept=%.4f",
                N, int(winner["segment_size"]), float(winner["hinge_accept"]),
                float(winner["tail_accept"]),
            )
        TSVWriter.upsert_pe_winners_row(
            self.bench_dir, N=N, residues_per_segment=int(winner["segment_size"])
        )
        return winner

    def _pick_pe1_winner(self, rows: List[Dict[str, str]]) -> Dict[str, object]:
        if not rows:
            return {"segment_size": 20, "per_sweep_ms": float("inf"),
                    "hinge_accept": 0.0, "tail_accept": 0.0,
                    "gpu_util_mean": float("nan"), "band_miss": True}
        def _in_band(x: float) -> bool:
            return self.ACCEPT_LOW <= x <= self.ACCEPT_HIGH
        parsed = []
        for r in rows:
            parsed.append({
                "segment_size": int(r["segment_size"]),
                "per_sweep_ms": float(r["per_sweep_ms"]),
                "hinge_accept": float(r["hinge_accept"]),
                "tail_accept": float(r["tail_accept"]),
                "gpu_util_mean": float(r.get("gpu_util_mean", "nan") or "nan"),
            })
        in_band = [p for p in parsed if _in_band(p["hinge_accept"]) and _in_band(p["tail_accept"])]
        if in_band:
            winner = min(in_band, key=lambda p: p["per_sweep_ms"])
            winner["band_miss"] = False
        else:
            winner = min(parsed, key=lambda p: p["per_sweep_ms"])
            winner["band_miss"] = True
        return winner

    # ------------------------------------------------------------------
    # PE2 — hinge-angle sweep per N
    # ------------------------------------------------------------------
    def run_pe2_for_N(self, N: int, residues_per_segment: int) -> Dict[str, object]:
        pe2_cfg = self.configs.benchmark.pe2
        n_chains = int(pe2_cfg.get("n_chains", 20))
        eq = int(self.pe_eq_override if self.pe_eq_override is not None
                 else pe2_cfg.get("eq_sweeps", 500))
        prod = int(self.pe_prod_override if self.pe_prod_override is not None
                   else pe2_cfg.get("prod_sweeps", 500))
        candidates: List[float] = list(pe2_cfg.get("candidates_rad", [math.pi / 2.0]))
        device = "cpu" if self.skip_cpu_pe else "gpu"

        _LOGGER.info("[BENCH-PE2-BEGIN] N=%d candidates_rad=%s device=%s",
                     N, candidates, device)
        for ang in candidates:
            cell_tag = f"PE2-N{N}-ang{ang:.4f}"
            argv = self._base_argv(
                device=device, algorithm="multistep", N=N, n_chains=n_chains,
                eq=eq, prod=prod, cell_tag=cell_tag, pe_mode="pe2",
                residues_per_segment=residues_per_segment,
                max_angle_hinge=ang, repeats=1, skip_timings_row=True,
            )
            t0 = time.perf_counter()
            try:
                self._invoke_cell(argv)
            except Exception as exc:
                _LOGGER.error("[BENCH-PE2-ERROR] N=%d ang=%.4f err=%r", N, ang, exc)
            _LOGGER.info("[BENCH-PE2-CAND-DONE] N=%d ang=%.4f wall=%.2fs",
                         N, ang, time.perf_counter() - t0)

        tsv_path = os.path.join(self.bench_dir, "pe2_angle_sweep.tsv")
        rows = self._read_pe_tsv(tsv_path, N)
        winner = self._pick_pe2_winner(rows)
        _LOGGER.info(
            "[BENCH-PE2-WINNER] N=%d MAX_ANGLE_HINGE=%.6f PerSweepMs=%.4f "
            "HingeAccept=%.4f TauIntR2=%.4f EffectiveSamplingRate=%.6f "
            "GpuUtilMean=%.4f BandMiss=%s Source=PE2",
            N, float(winner["max_angle_rad"]), float(winner["per_sweep_ms"]),
            float(winner["hinge_accept"]), float(winner["tau_int_R2"]),
            float(winner["effective_sampling_rate"]), float(winner["gpu_util_mean"]),
            winner["band_miss"],
        )
        if winner["band_miss"]:
            _LOGGER.warning(
                "[BENCH-PE2-BAND-MISS] N=%d ang=%.4f HingeAccept=%.4f",
                N, float(winner["max_angle_rad"]), float(winner["hinge_accept"]),
            )
        TSVWriter.upsert_pe_winners_row(
            self.bench_dir, N=N, max_angle_hinge=float(winner["max_angle_rad"])
        )
        return winner

    def _pick_pe2_winner(self, rows: List[Dict[str, str]]) -> Dict[str, object]:
        if not rows:
            return {"max_angle_rad": math.pi / 2.0, "per_sweep_ms": float("inf"),
                    "hinge_accept": 0.0, "tau_int_R2": 1.0,
                    "effective_sampling_rate": 0.0, "gpu_util_mean": float("nan"),
                    "band_miss": True}
        parsed = []
        for r in rows:
            parsed.append({
                "max_angle_rad": float(r["max_angle_rad"]),
                "per_sweep_ms": float(r["per_sweep_ms"]),
                "hinge_accept": float(r["hinge_accept"]),
                "tau_int_R2": float(r["tau_int_R2"]),
                "effective_sampling_rate": float(r["effective_sampling_rate"]),
                "gpu_util_mean": float(r.get("gpu_util_mean", "nan") or "nan"),
            })
        in_band = [p for p in parsed if self.ACCEPT_LOW <= p["hinge_accept"] <= self.ACCEPT_HIGH]
        if in_band:
            winner = max(in_band, key=lambda p: p["effective_sampling_rate"])
            winner["band_miss"] = False
        else:
            winner = max(parsed, key=lambda p: p["effective_sampling_rate"])
            winner["band_miss"] = True
        return winner

    # ------------------------------------------------------------------
    # PE3 — batch-size sweep per N
    # ------------------------------------------------------------------
    def run_pe3_for_N(self, N: int, residues_per_segment: int,
                       max_angle_hinge: float) -> Dict[str, object]:
        pe3_cfg = self.configs.benchmark.pe3 or {}
        n_chains = int(pe3_cfg.get("n_chains", 20))
        eq = int(self.pe3_eq_override if self.pe3_eq_override is not None
                 else pe3_cfg.get("eq_sweeps", 200))
        prod = int(self.pe3_prod_override if self.pe3_prod_override is not None
                   else pe3_cfg.get("prod_sweeps", 300))
        candidates: List[int] = list(pe3_cfg.get("candidates", [20, 50, 100, 200, 500]))
        floor = float(pe3_cfg.get("gpu_util_floor", self.GPU_UTIL_FLOOR))
        device = "gpu"  # PE3 targets GPU saturation

        _LOGGER.info("[BENCH-PE3-BEGIN] N=%d candidates=%s device=%s floor=%.2f",
                     N, candidates, device, floor)
        for bs in candidates:
            cell_tag = f"PE3-N{N}-bs{bs}"
            argv = self._base_argv(
                device=device, algorithm="multistep", N=N, n_chains=n_chains,
                eq=eq, prod=prod, cell_tag=cell_tag, pe_mode="pe3",
                residues_per_segment=residues_per_segment,
                max_angle_hinge=max_angle_hinge,
                batch_size=bs, repeats=1, skip_timings_row=True,
            )
            t0 = time.perf_counter()
            try:
                self._invoke_cell(argv)
            except Exception as exc:
                _LOGGER.error("[BENCH-PE3-ERROR] N=%d bs=%d err=%r", N, bs, exc)
            _LOGGER.info("[BENCH-PE3-CAND-DONE] N=%d bs=%d wall=%.2fs",
                         N, bs, time.perf_counter() - t0)

        tsv_path = os.path.join(self.bench_dir, "pe3_batch_sweep.tsv")
        rows = self._read_pe_tsv(tsv_path, N)
        winner, ceiling = self._pick_pe3_winner(rows, floor)
        _LOGGER.info(
            "[BENCH-PE3-WINNER] N=%d batch_size=%d PerSweepMs=%.4f "
            "GpuUtilMean=%.4f GpuUtilPeak=%.4f Ceiling=%s Source=PE3",
            N, int(winner["batch_size"]), float(winner["per_sweep_ms"]),
            float(winner["gpu_util_mean"]), float(winner["gpu_util_peak"]),
            ceiling,
        )
        if ceiling:
            _LOGGER.warning(
                "[BENCH-PE3-CEILING] N=%d BestGpuUtilMean=%.4f Threshold=%.2f",
                N, float(winner["gpu_util_mean"]), floor,
            )
        TSVWriter.upsert_pe_winners_row(
            self.bench_dir, N=N, batch_size=int(winner["batch_size"])
        )
        return winner

    def _pick_pe3_winner(self, rows: List[Dict[str, str]],
                         floor: float) -> Tuple[Dict[str, object], bool]:
        if not rows:
            return ({"batch_size": 100, "per_sweep_ms": float("inf"),
                     "gpu_util_mean": float("nan"), "gpu_util_peak": float("nan")},
                    True)
        parsed = []
        for r in rows:
            parsed.append({
                "batch_size": int(r["batch_size"]),
                "per_sweep_ms": float(r["per_sweep_ms"]),
                "gpu_util_mean": float(r.get("gpu_util_mean", "nan") or "nan"),
                "gpu_util_peak": float(r.get("gpu_util_peak", "nan") or "nan"),
            })
        above = [p for p in parsed
                 if math.isfinite(p["gpu_util_mean"]) and p["gpu_util_mean"] >= floor]
        if above:
            return (min(above, key=lambda p: p["per_sweep_ms"]), False)
        winner = min(parsed, key=lambda p: p["per_sweep_ms"])
        return winner, True

    # ------------------------------------------------------------------
    # B1 smoke + B2 dispatch trace
    # ------------------------------------------------------------------
    def run_b1_smoke(self) -> None:
        argv = self._base_argv(
            device="cpu", algorithm="conventional", N=25, n_chains=5,
            eq=5, prod=5, cell_tag="B1-smoke", repeats=1, b1_smoke=True,
            warmup_sweeps=0, skip_timings_row=True,
        )
        _LOGGER.info("[BENCH-B1-BEGIN] cell_tag=B1-smoke")
        self._invoke_cell(argv)

    def run_b2_dispatch(self) -> None:
        """Emit one [BENCH-B2] line per algorithm via a minimal cell."""
        for alg in self.ALGORITHMS:
            argv = self._base_argv(
                device="cpu", algorithm=alg, N=25, n_chains=2,
                eq=1, prod=1, cell_tag=f"B2-{alg}", repeats=1,
                warmup_sweeps=0, emit_b2=True, skip_timings_row=True,
            )
            _LOGGER.info("[BENCH-B2-BEGIN] algorithm=%s", alg)
            self._invoke_cell(argv)

    # ------------------------------------------------------------------
    # B3 matrix — 12 cells (N x algorithm x device) with --repeats=3
    # ------------------------------------------------------------------
    def run_b_matrix(self) -> None:
        winners = TSVWriter.read_pe_winners(self.bench_dir)
        _LOGGER.info("[BENCH-B-MATRIX-BEGIN] pe_winners=%s", winners)
        for N in self.N_VALUES:
            w = winners.get(N, {})
            seg = w.get("residues_per_segment")
            ang = w.get("max_angle_hinge")
            bs = w.get("batch_size")
            for alg in self.ALGORITHMS:
                for dev in self.DEVICES:
                    cell_tag = f"B-N{N}-{alg}-{dev}"
                    argv = self._base_argv(
                        device=dev, algorithm=alg, N=N,
                        n_chains=self.n_chains_bench,
                        eq=None, prod=None, cell_tag=cell_tag,
                        residues_per_segment=seg,
                        max_angle_hinge=ang,
                        batch_size=bs,
                        repeats=self.repeats,
                        timing_mode=self.b_timing_mode,
                        n_replicas=self.n_replicas,
                    )
                    _LOGGER.info("[BENCH-CELL-DISPATCH] %s argv=%s", cell_tag, argv)
                    t0 = time.perf_counter()
                    try:
                        self._invoke_cell(argv)
                    except Exception as exc:
                        _LOGGER.error("[BENCH-CELL-ERROR] %s err=%r", cell_tag, exc)
                    _LOGGER.info("[BENCH-CELL-DONE] %s wall=%.2fs",
                                 cell_tag, time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------
    def run_all(self) -> None:
        self.run_r7_cleanup()
        self.run_b1_smoke()
        self.run_b2_dispatch()
        if not self.skip_pe:
            for N in self.N_VALUES:
                pe1_win = self.run_pe1_for_N(N)
                pe2_win = self.run_pe2_for_N(N, int(pe1_win["segment_size"]))
                self.run_pe3_for_N(N, int(pe1_win["segment_size"]),
                                    float(pe2_win["max_angle_rad"]))
        if not self.skip_b_matrix:
            self.run_b_matrix()
        from rouse_model_python.src.libs.analysis.bench_analyzer import BenchAnalyzer
        BenchAnalyzer.run(self.bench_dir)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="BenchmarkOrchestrator — multi-cell driver (PE1/PE2/PE3 + 12-cell B-matrix).")
    p.add_argument("--output_dir", type=str, default="bench_output")
    p.add_argument("--phi", type=float, default=0.10)
    p.add_argument("--n_chains", type=int, default=300)
    p.add_argument("--warmup_sweeps", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_pe", action="store_true")
    p.add_argument("--skip_b_matrix", action="store_true")
    p.add_argument("--skip_cpu_pe", action="store_true")
    p.add_argument("--pe_eq", type=int, default=None)
    p.add_argument("--pe_prod", type=int, default=None)
    p.add_argument("--pe3_eq", type=int, default=None)
    p.add_argument("--pe3_prod", type=int, default=None)
    p.add_argument("--b_timing_mode", action="store_true",
                   help="Use benchmark.toml timing_sweep_schedule for B-cells instead of sweep_schedule.")
    p.add_argument("--n_replicas", type=int, default=1,
                   help="Number of independent chain replicas per B-cell (routed to BenchmarkApp via --n_replicas). "
                        "B1 smoke and B2 dispatch stay at R=1 — this flag only affects B-matrix cells.")
    p.add_argument("--only", type=str, default="",
                   help="One of: r7, b1, b2, pe1, pe2, pe3, b_matrix, analyze. Comma-separated to chain.")
    args = p.parse_args(argv)
    orch = BenchmarkOrchestrator(
        output_dir=args.output_dir, phi=args.phi,
        n_chains_bench=args.n_chains, warmup_sweeps=args.warmup_sweeps,
        repeats=args.repeats, seed=args.seed,
        skip_pe=args.skip_pe, skip_b_matrix=args.skip_b_matrix,
        skip_cpu_pe=args.skip_cpu_pe,
        pe_eq_override=args.pe_eq, pe_prod_override=args.pe_prod,
        pe3_eq_override=args.pe3_eq, pe3_prod_override=args.pe3_prod,
        b_timing_mode=args.b_timing_mode,
        n_replicas=args.n_replicas,
    )
    if args.only:
        orch._ensure_logging()
        steps = [s.strip() for s in args.only.split(",") if s.strip()]
        for step in steps:
            if step == "r7":
                orch.run_r7_cleanup()
            elif step == "b1":
                orch.run_b1_smoke()
            elif step == "b2":
                orch.run_b2_dispatch()
            elif step == "pe1":
                for N in orch.N_VALUES:
                    orch.run_pe1_for_N(N)
            elif step == "pe2":
                winners = TSVWriter.read_pe_winners(orch.bench_dir)
                for N in orch.N_VALUES:
                    seg = (winners.get(N) or {}).get("residues_per_segment") or 20
                    orch.run_pe2_for_N(N, int(seg))
            elif step == "pe3":
                winners = TSVWriter.read_pe_winners(orch.bench_dir)
                for N in orch.N_VALUES:
                    seg = (winners.get(N) or {}).get("residues_per_segment") or 20
                    ang = (winners.get(N) or {}).get("max_angle_hinge") or (math.pi / 2.0)
                    orch.run_pe3_for_N(N, int(seg), float(ang))
            elif step == "b_matrix":
                orch.run_b_matrix()
            elif step == "analyze":
                from rouse_model_python.src.libs.analysis.bench_analyzer import BenchAnalyzer
                BenchAnalyzer.run(orch.bench_dir)
            else:
                _LOGGER.error("Unknown --only step: %s", step)
                return 2
        return 0
    orch.run_all()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
