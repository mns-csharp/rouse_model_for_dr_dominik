"""BenchmarkApp — single-cell benchmark runner for the Rouse MC performance protocol.

Wraps the full pipeline: parse args, build SimulationConfig, run warmup + timed
sweeps with GPU sync bracketing AND GpuUtilPoller bracketing, write a row to
bench_timings.tsv, emit [BENCH-*] log lines. The actual analysis (B4-B7 gate
emission + plots) is invoked separately via BenchAnalyzer. Multi-cell driving
(PE1/PE2/PE3 per-N, 12-cell B-matrix) lives in BenchmarkOrchestrator.
"""

import argparse
import json
import logging
import math
import os
import statistics
import time
from typing import List, Tuple

import numpy as np
import torch

from rouse_model_python.src.libs.config.config_loader import ConfigLoader
from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.config.config_helpers import ConfigHelpers
from rouse_model_python.src.libs.execution.system_capabilities import SystemCapabilities
from rouse_model_python.src.libs.execution.execution_policy import ExecutionPolicy
from rouse_model_python.src.libs.io.bench_logging import BenchLogging
from rouse_model_python.src.libs.io.trajectory_writer import TrajectoryWriter
from rouse_model_python.src.libs.io.tsv_writer import TSVWriter
from rouse_model_python.src.libs.simulation.rouse_simulation import RouseSimulation
from rouse_model_python.src.libs.simulation.simulation_stats import SimulationStats
from rouse_model_python.src.libs.simulation.sweep_dispatcher import SweepDispatcher


class BenchmarkApp:
    def __init__(self, argv=None):
        self.argv = argv
        self.configs = ConfigLoader.load()
        self._logger = logging.getLogger("rouse_model_python.src.apps.benchmark")

    def build_arg_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Single-cell Rouse MC benchmark runner.")
        ExecutionPolicy.add_cli_args(p)
        p.add_argument("--output_dir", type=str, default="bench_output")
        p.add_argument("--N", type=int, default=25)
        p.add_argument("--n_chains", type=int,
                       default=self.configs.benchmark.cell_defaults.get("n_chains", 50))
        p.add_argument("--phi", type=float,
                       default=self.configs.benchmark.cell_defaults.get("phi", 0.035))
        p.add_argument("--eq_sweeps", type=int, default=None)
        p.add_argument("--prod_sweeps", type=int, default=None)
        p.add_argument("--residues_per_segment", type=int, default=None)
        p.add_argument("--max_angle_hinge", type=float, default=None)
        p.add_argument("--batch_size", type=int, default=None,
                       help="Multistep M / pivot batch size. Plumbed into "
                            "SimulationConfig.batch_size and consumed by "
                            "MultiStepMC. Default from benchmark.toml pe3 candidates.")
        p.add_argument("--warmup_sweeps", type=int,
                       default=self.configs.benchmark.cell_defaults.get("warmup_sweeps", 10))
        p.add_argument("--repeats", type=int,
                       default=self.configs.benchmark.cell_defaults.get("repeats", 1))
        p.add_argument("--cell_tag", type=str, default="cell")
        p.add_argument("--pe_mode", type=str, default="none",
                       choices=["none", "pe1", "pe2", "pe3"])
        p.add_argument("--b1_smoke", action="store_true", default=False)
        p.add_argument("--emit_b2", action="store_true", default=False,
                       help="Emit a [BENCH-B2] SweepPath line for dispatch verification.")
        p.add_argument("--skip_timings_row", action="store_true", default=False,
                       help="Skip writing a row to bench_timings.tsv. Used by PE "
                            "and smoke cells so the main B-matrix TSV stays clean.")
        p.add_argument("--conv_energy", type=str, default="cpu", choices=["cpu", "gpu"],
                       help="Energy primitive for conventional MC on GPU: 'cpu' keeps "
                            "the cell-list path (default, legacy); 'gpu' routes through "
                            "the shared fused kernel with B=1.")
        p.add_argument("--seed", type=int, default=self.configs.simulation.seed)
        p.add_argument("--timing_mode", action="store_true", default=False)
        p.add_argument("--n_replicas", type=int, default=1,
                       help="Number of independent chain replicas to run concurrently "
                            "on separate CUDA streams. >1 routes through MultiReplicaRunner "
                            "and reports per_sweep_s_effective = total_wall / (R * total_sweeps) "
                            "alongside the raw per_sweep_s. No replica exchange moves.")
        TrajectoryWriter.add_cli_args(p)
        return p

    def _compute_box_size(self, N, n_chains, phi):
        return ConfigHelpers.compute_box_size(N, n_chains, phi, self.configs.physics)

    def build_config(self, args) -> SimulationConfig:
        N = args.N
        n_chains = args.n_chains
        phi = args.phi
        device_str = "cuda" if args.device == "gpu" else args.device
        schedule = (self.configs.benchmark.timing_sweep_schedule
                    if args.timing_mode else self.configs.benchmark.sweep_schedule)
        if args.eq_sweeps is None or args.prod_sweeps is None:
            sched_eq, sched_prod = schedule.get(N, (500, 500))
        eq_sweeps = args.eq_sweeps if args.eq_sweeps is not None else sched_eq
        prod_sweeps = args.prod_sweeps if args.prod_sweeps is not None else sched_prod
        box = self._compute_box_size(N, n_chains, phi)
        cfg = SimulationConfig.from_defaults(
            N=N, n_chains=n_chains, eq_sweeps=eq_sweeps, prod_sweeps=prod_sweeps,
            box_size=box, device=device_str,
            physics=self.configs.physics,
            defaults=self.configs.simulation,
            target_phi=phi, algorithm=args.algorithm,
        )
        cfg.seed = args.seed
        if args.residues_per_segment is not None:
            cfg.residues_per_segment = args.residues_per_segment
        if args.max_angle_hinge is not None:
            cfg.max_angle_hinge = args.max_angle_hinge
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        cfg.use_batched_mode = (args.algorithm == "multistep" and device_str == "cuda")
        cfg.use_gpu_energy_path = (args.algorithm == "conventional"
                                   and device_str == "cuda"
                                   and args.conv_energy == "gpu")
        # Keep the acceptance loop and pivot phase on GPU whenever we're running
        # multistep on CUDA. Without these flags the fallback path in
        # MultiStepMC._process_batch does delta_e/correction/uniforms D2H per
        # batch (~20 syncs per sweep) and the pivot phase builds a CPU mirror
        # of state.positions — together ~70% of sweep wall time at small N.
        if cfg.use_batched_mode:
            cfg.accept_on_gpu = True
            cfg.pivot_on_gpu = True
        return cfg

    @staticmethod
    def _gpu_sync(device_str):
        if device_str == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _rate(num, denom):
        return float(num) / float(denom) if denom > 0 else 0.0

    @staticmethod
    def _integrated_autocorr_time(x):
        arr = np.asarray(x, dtype=np.float64)
        n = arr.size
        if n < 8:
            return 1.0
        arr = arr - arr.mean()
        var = float((arr * arr).mean())
        if var <= 0.0:
            return 1.0
        f = np.fft.fft(arr, n=2 * n)
        acf = np.fft.ifft(f * np.conj(f))[:n].real
        acf /= (var * n)
        tau = 1.0; c = 5.0
        for k in range(1, n):
            tau += 2.0 * acf[k]
            if k >= c * tau:
                break
        return max(1.0, float(tau))

    @staticmethod
    def _open_gpu_poller(device_str):
        """Return a GpuUtilPoller-like context on GPU; a no-op on CPU.

        The poller must bracket the SAME time.perf_counter() window that times
        per_sweep_s (per R5(h)) — this method is called immediately BEFORE t0.
        """
        class _NoopPoller:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def summary(self_inner):
                return {"mean_util": float("nan"), "peak_util": float("nan"),
                        "mean_mem_mib": float("nan"), "mean_power_w": float("nan")}

        if device_str != "cuda":
            return _NoopPoller()
        try:
            from rouse_model_python.src.libs.io.gpu_util_poller import GpuUtilPoller
            return GpuUtilPoller(interval_ms=100, live_print=False)
        except Exception:
            return _NoopPoller()

    def run_single(self, cfg: SimulationConfig, warmup_sweeps: int, device_str: str,
                   traj_writer: "TrajectoryWriter | None" = None,
                   n_replicas: int = 1):
        if traj_writer is not None:
            self._logger.info(
                "[BENCH-TRAJ] trajectory writer attached; per-sweep timings include trajectory I/O overhead")
        if n_replicas > 1:
            return self._run_single_multi_replica(cfg, warmup_sweeps, device_str,
                                                   traj_writer, n_replicas)
        sim = RouseSimulation(cfg, snapshot_collector=traj_writer)
        sim.initialize()
        if warmup_sweeps > 0:
            sweep_fn, _ = SweepDispatcher.resolve(cfg)
            for _ in range(warmup_sweeps):
                sweep_fn(sim.state, sim.energy_comp, sim.gen, cfg, sim.stats)
            self._gpu_sync(device_str)
            sim.stats = SimulationStats()

        poller = self._open_gpu_poller(device_str)
        with poller:
            self._gpu_sync(device_str)
            t0 = time.perf_counter()
            sim.run_equilibration()
            sim.run_production()
            self._gpu_sync(device_str)
            t1 = time.perf_counter()
        summary = poller.summary()
        gpu_util_mean = summary.get("mean_util", float("nan"))
        gpu_util_peak = summary.get("peak_util", float("nan"))
        if isinstance(gpu_util_mean, (int, float)) and math.isfinite(gpu_util_mean):
            gpu_util_mean = float(gpu_util_mean) / 100.0  # NVML reports 0-100, normalize to [0,1]
        if isinstance(gpu_util_peak, (int, float)) and math.isfinite(gpu_util_peak):
            gpu_util_peak = float(gpu_util_peak) / 100.0

        total = t1 - t0
        total_sweeps = cfg.eq_sweeps + cfg.prod_sweeps
        per = total / max(1, total_sweeps)
        obs = {
            "mean_R2": float(sim.final_R2.mean().item()) if sim.final_R2 is not None else float("nan"),
            "mean_Rg2": float(sim.final_Rg2.mean().item()) if sim.final_Rg2 is not None else float("nan"),
            "hinge_accept": self._rate(sim.stats.hinge_accepted, sim.stats.hinge_attempted),
            "tail_accept": self._rate(sim.stats.n_tail_accepted + sim.stats.c_tail_accepted,
                                      sim.stats.n_tail_attempted + sim.stats.c_tail_attempted),
            "pivot_accept": self._rate(sim.stats.pivot_accepted, sim.stats.pivot_attempted),
            "gpu_util_mean": gpu_util_mean,
            "gpu_util_peak": gpu_util_peak,
            "r2_series": getattr(sim.sweep_tracker, "records", []),
            "n_replicas": 1,
            "per_sweep_s_effective": per,
        }
        return total, per, obs

    def _run_single_multi_replica(self, cfg: SimulationConfig, warmup_sweeps: int,
                                   device_str: str, traj_writer, n_replicas: int):
        from rouse_model_python.src.libs.simulation.multi_replica_runner import MultiReplicaRunner
        runner = MultiReplicaRunner(cfg, n_replicas=n_replicas,
                                     snapshot_collector=traj_writer)
        runner.initialize()
        if warmup_sweeps > 0:
            sweep_fn = runner._sweep_fn
            for _ in range(warmup_sweeps):
                for sim in runner.replicas:
                    sweep_fn(sim.state, sim.energy_comp, sim.gen,
                             sim.cfg, sim.stats, skip_pivot=False)
            self._gpu_sync(device_str)
            from rouse_model_python.src.libs.simulation.simulation_stats import SimulationStats as _SS
            for sim in runner.replicas:
                sim.stats = _SS()

        poller = self._open_gpu_poller(device_str)
        with poller:
            self._gpu_sync(device_str)
            t0 = time.perf_counter()
            runner.run_equilibration()
            runner.run_production()
            self._gpu_sync(device_str)
            t1 = time.perf_counter()
        summary = poller.summary()
        gpu_util_mean = summary.get("mean_util", float("nan"))
        gpu_util_peak = summary.get("peak_util", float("nan"))
        if isinstance(gpu_util_mean, (int, float)) and math.isfinite(gpu_util_mean):
            gpu_util_mean = float(gpu_util_mean) / 100.0
        if isinstance(gpu_util_peak, (int, float)) and math.isfinite(gpu_util_peak):
            gpu_util_peak = float(gpu_util_peak) / 100.0

        total = t1 - t0
        total_sweeps = cfg.eq_sweeps + cfg.prod_sweeps
        per = total / max(1, total_sweeps)
        per_effective = total / max(1, n_replicas * total_sweeps)
        mean_R2, mean_Rg2 = runner.aggregate_final_observables()
        rates = runner.aggregate_stats()
        obs = {
            "mean_R2": mean_R2,
            "mean_Rg2": mean_Rg2,
            "hinge_accept": rates["hinge_rate"],
            "tail_accept": rates["tail_rate"],
            "pivot_accept": rates["pivot_rate"],
            "gpu_util_mean": gpu_util_mean,
            "gpu_util_peak": gpu_util_peak,
            "r2_series": [],
            "n_replicas": n_replicas,
            "per_sweep_s_effective": per_effective,
        }
        self._logger.info(
            "[BENCH-MR] Replicas=%d TotalWallSeconds=%.6f PerSweepSeconds=%.6f "
            "PerSweepSecondsEffective=%.6f HingeAccept=%.4f TailAccept=%.4f "
            "PivotAccept=%.4f MeanR2=%.4f MeanRg2=%.4f GpuUtilMean=%.4f",
            n_replicas, total, per, per_effective,
            rates["hinge_rate"], rates["tail_rate"], rates["pivot_rate"],
            mean_R2, mean_Rg2, gpu_util_mean,
        )
        return total, per, obs

    def run(self) -> int:
        args = self.build_arg_parser().parse_args(self.argv)
        caps = SystemCapabilities.query()
        is_parallel = args.is_parallel.lower() == "true"
        policy = ExecutionPolicy.resolve(args.device, is_parallel, caps, force=args.force)
        out_root = args.output_dir
        bench_dir = os.path.join(out_root, "06_benchmarks")
        TSVWriter.ensure_dir(bench_dir)
        BenchLogging.configure(bench_dir)
        logger = self._logger

        logger.info(
            "[BENCH-CELL-BEGIN] cell_tag=%s algorithm=%s device=%s is_parallel=%s N=%d n_chains=%d phi=%.4f",
            args.cell_tag, args.algorithm, args.device, is_parallel,
            args.N, args.n_chains, args.phi,
        )

        cfg = self.build_config(args)
        device_str = cfg.device
        logger.info(
            "[BENCH-CFG] N=%d n_chains=%d eq=%d prod=%d box=%.2f residues_per_segment=%d max_angle_hinge=%.4f batch_size=%d batched=%s",
            cfg.N, cfg.n_chains, cfg.eq_sweeps, cfg.prod_sweeps, cfg.box_size,
            cfg.residues_per_segment, cfg.max_angle_hinge, cfg.batch_size,
            cfg.use_batched_mode,
        )

        if args.emit_b2:
            _, sweep_path_str = SweepDispatcher.resolve(cfg)
            logger.info(
                "[BENCH-B2] Algorithm=%s SweepPath=%s",
                cfg.algorithm, sweep_path_str,
            )

        traj_dir = os.path.join(out_root, "07_trajectories")
        totals, pers, last_obs = [], [], {}
        for run_i in range(args.repeats):
            writer = (TrajectoryWriter.from_args(args, cfg, traj_dir, run_label=args.cell_tag)
                      if run_i == 0 else None)
            total, per, obs = self.run_single(cfg, args.warmup_sweeps, device_str,
                                              traj_writer=writer,
                                              n_replicas=args.n_replicas)
            if writer is not None:
                writer.close()
            totals.append(total); pers.append(per); last_obs = obs
            logger.info(
                "[BENCH-RUN] cell_tag=%s run=%d/%d total_wall_s=%.6f per_sweep_s=%.6f hinge=%.4f tail=%.4f pivot=%.4f R2=%.2f Rg2=%.2f gpu_util=%.4f",
                args.cell_tag, run_i + 1, args.repeats, total, per,
                obs["hinge_accept"], obs["tail_accept"], obs["pivot_accept"],
                obs["mean_R2"], obs["mean_Rg2"], obs["gpu_util_mean"],
            )

        median_total = statistics.median(totals)
        median_per = statistics.median(pers)

        logger.info(
            "[BENCH-B3] N=%d Algorithm=%s Device=%s ChainsProduction=%d SweepsProduction=%d TotalSweepSeconds=%.6f MeanPerSweepSeconds=%.6f RunID=median-of-%d",
            cfg.N, cfg.algorithm, args.device, cfg.n_chains, cfg.prod_sweeps,
            median_total, median_per, args.repeats,
        )

        if not args.skip_timings_row:
            TSVWriter.write_bench_timings_row(
                bench_dir,
                N=cfg.N, algorithm=cfg.algorithm, device=args.device,
                n_chains=cfg.n_chains, eq_sweeps=cfg.eq_sweeps, prod_sweeps=cfg.prod_sweeps,
                total_wall_s=median_total, per_sweep_s=median_per,
                hinge_accept=last_obs["hinge_accept"],
                tail_accept=last_obs["tail_accept"],
                pivot_accept=last_obs["pivot_accept"],
                segment_size=cfg.residues_per_segment,
                max_angle_hinge=cfg.max_angle_hinge,
                batch_size=cfg.batch_size,
                gpu_util_mean=last_obs["gpu_util_mean"],
                run_id=f"{args.cell_tag}-median-of-{args.repeats}",
                warmup_sweeps=args.warmup_sweeps,
                wall_clock_median_of_3=(args.repeats >= 3),
                n_replicas=last_obs.get("n_replicas", 1),
                per_sweep_s_effective=last_obs.get("per_sweep_s_effective", median_per),
            )

        if args.pe_mode == "pe1":
            logger.info(
                "[BENCH-PE1] N=%d Segment=%d PerSweepMs=%.4f HingeAccept=%.4f TailAccept=%.4f PivotAccept=%.4f GpuUtilMean=%.4f MeanR2=%.4f MeanRg2=%.4f",
                cfg.N, cfg.residues_per_segment, median_per * 1000.0,
                last_obs["hinge_accept"], last_obs["tail_accept"],
                last_obs["pivot_accept"], last_obs["gpu_util_mean"],
                last_obs["mean_R2"], last_obs["mean_Rg2"],
            )
            TSVWriter.append_row(
                os.path.join(bench_dir, "pe1_segment_sweep.tsv"),
                header=["N", "segment_size", "per_sweep_ms", "hinge_accept",
                        "tail_accept", "pivot_accept", "gpu_util_mean",
                        "mean_R2", "mean_Rg2"],
                row=[cfg.N, cfg.residues_per_segment, median_per * 1000.0,
                     last_obs["hinge_accept"], last_obs["tail_accept"],
                     last_obs["pivot_accept"], last_obs["gpu_util_mean"],
                     last_obs["mean_R2"], last_obs["mean_Rg2"]],
            )
        elif args.pe_mode == "pe2":
            r2_series = [float(rec[2]) for rec in last_obs.get("r2_series", [])
                         if len(rec) >= 3 and rec[1] == "production"]
            tau = self._integrated_autocorr_time([v for v in r2_series if math.isfinite(v)])
            eff = 1.0 / (tau * median_per) if (tau > 0 and median_per > 0) else 0.0
            logger.info(
                "[BENCH-PE2] N=%d MaxAngleRad=%.6f PerSweepMs=%.4f HingeAccept=%.4f TauIntR2=%.4f EffectiveSamplingRate=%.6f GpuUtilMean=%.4f",
                cfg.N, cfg.max_angle_hinge, median_per * 1000.0,
                last_obs["hinge_accept"], tau, eff, last_obs["gpu_util_mean"],
            )
            TSVWriter.append_row(
                os.path.join(bench_dir, "pe2_angle_sweep.tsv"),
                header=["N", "max_angle_rad", "per_sweep_ms", "hinge_accept",
                        "tau_int_R2", "effective_sampling_rate", "gpu_util_mean"],
                row=[cfg.N, cfg.max_angle_hinge, median_per * 1000.0,
                     last_obs["hinge_accept"], tau, eff,
                     last_obs["gpu_util_mean"]],
            )
        elif args.pe_mode == "pe3":
            logger.info(
                "[BENCH-PE3] N=%d BatchSize=%d PerSweepMs=%.4f GpuUtilMean=%.4f GpuUtilPeak=%.4f HingeAccept=%.4f",
                cfg.N, cfg.batch_size, median_per * 1000.0,
                last_obs["gpu_util_mean"], last_obs["gpu_util_peak"],
                last_obs["hinge_accept"],
            )
            TSVWriter.append_row(
                os.path.join(bench_dir, "pe3_batch_sweep.tsv"),
                header=["N", "batch_size", "per_sweep_ms", "gpu_util_mean",
                        "gpu_util_peak", "hinge_accept"],
                row=[cfg.N, cfg.batch_size, median_per * 1000.0,
                     last_obs["gpu_util_mean"], last_obs["gpu_util_peak"],
                     last_obs["hinge_accept"]],
            )

        if args.b1_smoke:
            import_ok = True
            try:
                from rouse_model_python.src.libs.algorithms.conventional_mc import ConventionalMC  # noqa: F401
            except Exception:
                import_ok = False
            smoke_ok = median_total > 0.0 and import_ok
            logger.info(
                "[BENCH-B1] ConventionalMCImport=%s SmokeRun=%s SmokeWallSeconds=%.6f",
                "pass" if import_ok else "fail",
                "pass" if smoke_ok else "fail",
                median_total,
            )

        logger.info(
            "[BENCH-CELL-END] cell_tag=%s median_total_wall_s=%.6f median_per_sweep_s=%.6f",
            args.cell_tag, median_total, median_per,
        )

        print(json.dumps({
            "cell_tag": args.cell_tag,
            "N": cfg.N, "algorithm": cfg.algorithm, "device": args.device,
            "n_chains": cfg.n_chains, "eq": cfg.eq_sweeps, "prod": cfg.prod_sweeps,
            "total_wall_s": median_total, "per_sweep_s": median_per,
            "hinge": last_obs["hinge_accept"], "tail": last_obs["tail_accept"],
            "pivot": last_obs["pivot_accept"],
            "R2": last_obs["mean_R2"], "Rg2": last_obs["mean_Rg2"],
            "gpu_util_mean": last_obs["gpu_util_mean"],
            "batch_size": cfg.batch_size,
        }, indent=2), flush=True)
        return 0
