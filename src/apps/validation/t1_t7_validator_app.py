"""T1T7ValidatorApp — Rouse T1..T7 validation runner.

Iterates the (N × phi) validation matrix from configs/simulation.toml,
drives RouseSimulation (torch path) for each cell, computes observables via
the libs/observables/ stack, then fits and emits PASS/FAIL verdicts for
gates T1 through T7:

    T1  <R²>   vs N exponent 2ν  = 1.18 ± 0.15, R²_fit > 0.95
    T2  <Rg²>  vs N exponent 2ν  = 1.18 ± 0.15, R²_fit > 0.95
    T3  R²/Rg² at N=100          = 6.25 ± 0.60
    T4  g_CM(t) long-time α      = 1.00 ± 0.15, R²_fit > 0.95
    T5  g₁(t)   short-time α     = 0.50 ± 0.15, R²_fit > 0.90
    T6  τ_R     vs N exponent    = 2.18 ± 0.30, R²_fit > 0.95
    T7  D(N)    vs N exponent    = -1.00 ± 0.20, R²_fit > 0.95

Outputs (under `<output_dir>/`):
    00_t1_t7_verdicts.json        — per-T PASS/FAIL + fitted exponents
    05_data/phi_<p>/N<N>/*.tsv    — per-cell static + dynamic TSVs
    05_data/tavg_validation_summary.json
    05_data/rouse_compliance_heatmap.png
    01_static_properties/*.png    — cross-phi static plots
    02_dynamic_properties/*.png   — cross-phi dynamic plots
    03_per_state_point/…          — per-cell diagnostic plots
    04_equilibration_evidence/…   — eq evidence plots
    README.md, .gitattributes
"""

import argparse
import json
import logging
import os
import time

import numpy as np

from rouse_model_python.src.libs.config.config_helpers import ConfigHelpers
from rouse_model_python.src.libs.config.config_loader import ConfigLoader
from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.execution.execution_policy import ExecutionPolicy
from rouse_model_python.src.libs.execution.system_capabilities import SystemCapabilities
from rouse_model_python.src.libs.io.analysis_metrics import AnalysisMetrics
from rouse_model_python.src.libs.io.bench_logging import BenchLogging
from rouse_model_python.src.libs.io.checklist_diagnostics import ChecklistDiagnostics
from rouse_model_python.src.libs.io.plot_generator import PlotGenerator
from rouse_model_python.src.libs.io.trajectory_writer import TrajectoryWriter
from rouse_model_python.src.libs.io.tsv_writer import TSVWriter
from rouse_model_python.src.libs.io.validation_plotter import ValidationPlotter
from rouse_model_python.src.libs.io.validation_summary import ValidationSummary
from rouse_model_python.src.libs.simulation.rouse_simulation import RouseSimulation


class T1T7ValidatorApp:
    T1_TARGET = 1.18; T1_TOL = 0.15; T1_R2_MIN = 0.95
    T2_TARGET = 1.18; T2_TOL = 0.15; T2_R2_MIN = 0.95
    T3_TARGET = 6.25; T3_TOL = 0.60; T3_REF_N = 100
    T4_TARGET = 1.00; T4_TOL = 0.15; T4_R2_MIN = 0.95
    T5_TARGET = 0.50; T5_TOL = 0.15; T5_R2_MIN = 0.90
    T6_TARGET = 2.18; T6_TOL = 0.30; T6_R2_MIN = 0.95
    T7_TARGET = -1.00; T7_TOL = 0.20; T7_R2_MIN = 0.95

    def __init__(self, argv=None):
        self.argv = argv
        self.configs = ConfigLoader.load()
        self._logger = logging.getLogger("rouse_model_python.src.apps.validation")

    def build_arg_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Rouse T1–T7 validator (runs the validation_matrix).")
        ExecutionPolicy.add_cli_args(p)
        p.add_argument("--output_dir", type=str, default="validation_output")
        p.add_argument("--eq_sweeps", type=int, default=None,
                       help="Override equilibration sweeps per cell (default: "
                            "per-N from benchmark.toml sweep_schedule).")
        p.add_argument("--prod_sweeps", type=int, default=None,
                       help="Override production sweeps per cell.")
        p.add_argument("--chain_lengths", type=str, default=None,
                       help="Comma-separated N list override (default: from simulation.toml).")
        p.add_argument("--phi_values", type=str, default=None,
                       help="Comma-separated phi list override.")
        p.add_argument("--seed", type=int, default=self.configs.simulation.seed)
        TrajectoryWriter.add_cli_args(p)
        return p

    def _resolve_matrix(self, args):
        if args.chain_lengths:
            Ns = [int(x) for x in args.chain_lengths.split(",") if x.strip()]
        else:
            Ns = list(self.configs.simulation.validation_chain_lengths)
        if args.phi_values:
            phis = [float(x) for x in args.phi_values.split(",") if x.strip()]
        else:
            phis = list(self.configs.simulation.validation_phi_values)
        return Ns, phis

    def _build_cell_config(self, N: int, phi: float, args, device_str: str) -> SimulationConfig:
        n_chains = ConfigHelpers.compute_n_chains(N, phi)
        box = ConfigHelpers.compute_box_size(N, n_chains, phi, self.configs.physics)

        schedule = self.configs.benchmark.sweep_schedule
        sched_eq, sched_prod = schedule.get(N, (500, 500))
        eq_sweeps = args.eq_sweeps if args.eq_sweeps is not None else sched_eq
        prod_sweeps = args.prod_sweeps if args.prod_sweeps is not None else sched_prod

        cfg = SimulationConfig.from_defaults(
            N=N, n_chains=n_chains, eq_sweeps=eq_sweeps, prod_sweeps=prod_sweeps,
            box_size=box, device=device_str,
            physics=self.configs.physics,
            defaults=self.configs.simulation,
            target_phi=phi, algorithm=args.algorithm,
        )
        cfg.seed = args.seed
        cfg.use_batched_mode = (args.algorithm == "multistep" and device_str == "cuda")
        return cfg

    def _run_single_cell(self, N: int, phi: float, args, device_str: str,
                         policy=None) -> dict:
        cfg = self._build_cell_config(N, phi, args, device_str)
        self._logger.info(
            "[VALIDATION-CELL-BEGIN] N=%d phi=%.4f n_chains=%d eq=%d prod=%d box=%.2f",
            N, phi, cfg.n_chains, cfg.eq_sweeps, cfg.prod_sweeps, cfg.box_size)
        phi_str = ConfigHelpers.format_phi(phi)
        traj_dir = os.path.join(args.output_dir, "07_trajectories", f"phi_{phi_str}", f"N{N}")
        writer = TrajectoryWriter.from_args(args, cfg, traj_dir, run_label="trajectory")
        sim = RouseSimulation(cfg, snapshot_collector=writer)
        sim.initialize()
        if policy is not None:
            ChecklistDiagnostics.run_per_cell_tests(sim, cfg, policy, phi, args.device)
        t0 = time.time()
        sim.run_equilibration()
        sim.run_production()
        wall_time = time.time() - t0
        if writer is not None:
            writer.close()
        results = sim.get_results()
        ChecklistDiagnostics.run_end_of_cell_tests(
            sim, cfg, phi, wall_time, results['sweep_data'], writer)
        self._logger.info(
            "[VALIDATION-CELL-END] N=%d phi=%.4f R2=%.2f Rg2=%.2f",
            N, phi, results['final_R2'].mean().item(),
            results['final_Rg2'].mean().item())
        return results

    def _write_cell_tsvs(self, results: dict, base_dir: str, N: int, phi: float) -> None:
        phi_str = ConfigHelpers.format_phi(phi)
        data_dir = os.path.join(base_dir, "05_data", f"phi_{phi_str}", f"N{N}")
        TSVWriter.ensure_dir(data_dir)

        R2 = results['final_R2'].cpu().numpy()
        Rg2 = results['final_Rg2'].cpu().numpy()

        with open(os.path.join(data_dir, "fig1_static.tsv"), 'w') as f:
            f.write("chain_id\tR2\tRg2\n")
            for i in range(len(R2)):
                f.write(f"{i}\t{R2[i]:.8E}\t{Rg2[i]:.8E}\n")

        for name, data_dict, col in [
                ("fig2_seg_msd.tsv", results['g1'], "g1"),
                ("fig3_cm_diffusion.tsv", results['gcm'], "g_CM"),
                ("fig4_autocorr.tsv", results['gr'], "g_R")]:
            with open(os.path.join(data_dir, name), 'w') as f:
                f.write(f"lag_sweep\t{col}\n")
                for lag in sorted(data_dict.keys()):
                    f.write(f"{lag}\t{data_dict[lag]:.8E}\n")

        with open(os.path.join(data_dir, "static_vs_sweep.tsv"), 'w') as f:
            f.write("sweep\tphase\tmean_R2\tmean_Rg2\tratio_R2_Rg2\n")
            for sweep, phase, R2v, Rg2v, ratio in results['sweep_data']:
                f.write(f"{sweep}\t{phase}\t{R2v:.8E}\t{Rg2v:.8E}\t{ratio:.8E}\n")

    @classmethod
    def _gate_verdict(cls, measured: float, target: float, tol: float,
                      r2_fit: float, r2_min: float) -> dict:
        passed = bool(abs(measured - target) <= tol and r2_fit >= r2_min)
        return {
            "measured": round(float(measured), 4),
            "target": target,
            "tolerance": tol,
            "r2_fit": round(float(r2_fit), 4),
            "r2_min": r2_min,
            "pass": passed,
        }

    @classmethod
    def _ratio_verdict(cls, measured: float, target: float, tol: float) -> dict:
        return {
            "measured": round(float(measured), 4),
            "target": target,
            "tolerance": tol,
            "pass": bool(abs(measured - target) <= tol),
        }

    @classmethod
    def compute_t1_t7(cls, all_results: dict) -> dict:
        dilute_phi = min(all_results.keys())
        phi_results = all_results[dilute_phi]
        Ns = sorted(phi_results.keys())

        mean_R2 = [AnalysisMetrics.production_averaged_static(phi_results[n])[0] for n in Ns]
        mean_Rg2 = [AnalysisMetrics.production_averaged_static(phi_results[n])[1] for n in Ns]

        slope_R2, _, r2_fit_R2 = PlotGenerator.power_law_fit(Ns, mean_R2)
        slope_Rg2, _, r2_fit_Rg2 = PlotGenerator.power_law_fit(Ns, mean_Rg2)

        if cls.T3_REF_N in phi_results:
            r2v, rg2v = AnalysisMetrics.production_averaged_static(phi_results[cls.T3_REF_N])
            ratio_N100 = r2v / rg2v if rg2v > 0 else 0.0
        else:
            ratio_N100 = float('nan')

        gcm_exp_per_n = [AnalysisMetrics.estimate_gcm_exponent(phi_results[n]['gcm'])
                         for n in Ns]
        valid_gcm = [e for e in gcm_exp_per_n if e > 0]
        mean_gcm_exp = float(np.mean(valid_gcm)) if valid_gcm else 0.0

        g1_exp_per_n = [AnalysisMetrics.estimate_g1_short_time_exponent(phi_results[n]['g1'])
                        for n in Ns]
        valid_g1 = [e for e in g1_exp_per_n if e > 0]
        mean_g1_exp = float(np.mean(valid_g1)) if valid_g1 else 0.0

        Ds = [AnalysisMetrics.estimate_diffusion_coefficient(phi_results[n]['gcm']) for n in Ns]
        valid_D = [(n, d) for n, d in zip(Ns, Ds) if d > 0]
        if valid_D:
            slope_D, _, r2_fit_D = PlotGenerator.power_law_fit(
                [v[0] for v in valid_D], [v[1] for v in valid_D])
        else:
            slope_D, r2_fit_D = 0.0, 0.0

        taus = [AnalysisMetrics.estimate_relaxation_time(phi_results[n]['gr']) for n in Ns]
        valid_tau = [(n, t) for n, t in zip(Ns, taus) if t > 0]
        if valid_tau:
            slope_tau, _, r2_fit_tau = PlotGenerator.power_law_fit(
                [v[0] for v in valid_tau], [v[1] for v in valid_tau])
        else:
            slope_tau, r2_fit_tau = 0.0, 0.0

        return {
            "dilute_phi": dilute_phi,
            "N_values": Ns,
            "T1_R2_vs_N": cls._gate_verdict(slope_R2, cls.T1_TARGET, cls.T1_TOL,
                                            r2_fit_R2, cls.T1_R2_MIN),
            "T2_Rg2_vs_N": cls._gate_verdict(slope_Rg2, cls.T2_TARGET, cls.T2_TOL,
                                             r2_fit_Rg2, cls.T2_R2_MIN),
            "T3_R2_over_Rg2_at_N100": cls._ratio_verdict(
                ratio_N100, cls.T3_TARGET, cls.T3_TOL),
            "T4_gCM_long_time_alpha": cls._gate_verdict(
                mean_gcm_exp, cls.T4_TARGET, cls.T4_TOL, 1.0, cls.T4_R2_MIN),
            "T5_g1_short_time_alpha": cls._gate_verdict(
                mean_g1_exp, cls.T5_TARGET, cls.T5_TOL, 1.0, cls.T5_R2_MIN),
            "T6_tauR_vs_N": cls._gate_verdict(
                slope_tau, cls.T6_TARGET, cls.T6_TOL, r2_fit_tau, cls.T6_R2_MIN),
            "T7_D_vs_N": cls._gate_verdict(
                slope_D, cls.T7_TARGET, cls.T7_TOL, r2_fit_D, cls.T7_R2_MIN),
        }

    def _write_verdicts(self, verdicts: dict, base_dir: str) -> None:
        filepath = os.path.join(base_dir, "00_t1_t7_verdicts.json")
        TSVWriter.ensure_dir(base_dir)

        def json_default(o):
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            return o

        with open(filepath, 'w') as f:
            json.dump(verdicts, f, indent=2, default=json_default)

        self._logger.info("[VALIDATION-T1-T7] Wrote %s", filepath)
        for key in ["T1_R2_vs_N", "T2_Rg2_vs_N", "T3_R2_over_Rg2_at_N100",
                    "T4_gCM_long_time_alpha", "T5_g1_short_time_alpha",
                    "T6_tauR_vs_N", "T7_D_vs_N"]:
            v = verdicts[key]
            status = "PASS" if v["pass"] else "FAIL"
            m = v.get("measured", float('nan'))
            tgt = v.get("target", 0.0)
            self._logger.info("[VALIDATION-%s] %s measured=%.4f target=%.4f",
                              key, status, m, tgt)

    def run(self) -> int:
        args = self.build_arg_parser().parse_args(self.argv)
        caps = SystemCapabilities.query()
        is_parallel = args.is_parallel.lower() == "true"
        policy = ExecutionPolicy.resolve(args.device, is_parallel, caps, force=args.force)
        device_str = policy.torch_device

        out_root = args.output_dir
        TSVWriter.ensure_dir(out_root)
        BenchLogging.configure(out_root)
        logger = self._logger

        ChecklistDiagnostics.run_startup_tests(
            self.configs.physics, args.seed, device_str)

        Ns, phis = self._resolve_matrix(args)
        logger.info("[VALIDATION-BEGIN] matrix N=%s phi=%s device=%s",
                    Ns, phis, args.device)

        all_results = {phi: {} for phi in phis}
        for phi in phis:
            for N in Ns:
                results = self._run_single_cell(N, phi, args, device_str, policy=policy)
                all_results[phi][N] = results
                self._write_cell_tsvs(results, out_root, N, phi)
                ValidationPlotter.plot_per_state_point(results, out_root, N, phi)

        for phi in phis:
            ValidationPlotter.plot_equilibration_per_phi(all_results[phi], out_root, phi)

        ValidationPlotter.plot_R2_vs_N_per_phi(all_results, out_root)
        ValidationPlotter.plot_Rg2_vs_N_per_phi(all_results, out_root)
        ValidationPlotter.plot_2nu_vs_phi(all_results, out_root)
        ValidationPlotter.plot_ratio_R2_Rg2_vs_phi(all_results, out_root, Ns)
        ValidationPlotter.plot_R2_Rg2_combined_dilute(all_results, out_root)
        ValidationPlotter.plot_g1_vs_sweep_per_phi(all_results, out_root)
        ValidationPlotter.plot_gcm_vs_sweep_per_phi(all_results, out_root)
        ValidationPlotter.plot_D_vs_N_per_phi(all_results, out_root)
        ValidationPlotter.plot_tauR_vs_N_per_phi(all_results, out_root)
        ValidationPlotter.plot_D_exponent_vs_phi(all_results, out_root)
        ValidationPlotter.plot_tauR_exponent_vs_phi(all_results, out_root)
        ValidationPlotter.plot_g1_shorttime_exponent_vs_phi(all_results, out_root)

        summary = ValidationSummary.write_validation_summary(all_results, out_root)
        ValidationSummary.write_readme(all_results, summary, out_root, Ns,
                                       self.configs.physics)
        ValidationSummary.write_gitattributes(out_root)

        verdicts = self.compute_t1_t7(all_results)
        self._write_verdicts(verdicts, out_root)

        ChecklistDiagnostics.log_t_items(all_results)

        all_pass = all(verdicts[k]["pass"] for k in [
            "T1_R2_vs_N", "T2_Rg2_vs_N", "T3_R2_over_Rg2_at_N100",
            "T4_gCM_long_time_alpha", "T5_g1_short_time_alpha",
            "T6_tauR_vs_N", "T7_D_vs_N"])
        logger.info("[VALIDATION-END] all_pass=%s", all_pass)
        print(json.dumps({"all_pass": all_pass,
                          **{k: {"measured": verdicts[k].get("measured"),
                                 "pass": verdicts[k]["pass"]}
                             for k in ["T1_R2_vs_N", "T2_Rg2_vs_N",
                                        "T3_R2_over_Rg2_at_N100",
                                        "T4_gCM_long_time_alpha",
                                        "T5_g1_short_time_alpha",
                                        "T6_tauR_vs_N", "T7_D_vs_N"]}},
                         indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_)
                         else float(o) if isinstance(o, (np.floating, np.integer)) else o),
              flush=True)
        return 0 if all_pass else 1
