"""ValidationPlotter — per-state-point and cross-phi plots for Rouse validation.

All methods are @classmethod so plots can be generated without constructing the
class. Uses PlotGenerator for figure primitives and AnalysisMetrics for
scaling estimators.
"""

import os

import numpy as np

from rouse_model_python.src.libs.config.config_helpers import ConfigHelpers
from rouse_model_python.src.libs.io.analysis_metrics import AnalysisMetrics
from rouse_model_python.src.libs.io.plot_generator import PlotGenerator


class ValidationPlotter:
    @classmethod
    def plot_per_state_point(cls, results: dict, base_dir: str, N: int, phi: float) -> None:
        phi_str = ConfigHelpers.format_phi(phi)
        plot_dir = os.path.join(base_dir, "03_per_state_point", f"phi_{phi_str}", f"N{N}")
        PlotGenerator.ensure_dir(plot_dir)

        sweep_data = results['sweep_data']
        title_suffix = f"(N={N}, phi={phi_str})"

        fig, ax = PlotGenerator.setup_plot(
            "MC Sweep", "<R2> (A^2)", f"R2 vs MC Sweep {title_suffix}", loglog=False)
        if sweep_data:
            sweeps = [d[0] for d in sweep_data]
            R2s = [d[2] for d in sweep_data]
            ax.plot(sweeps, R2s, '-', color='C0', linewidth=1)
        PlotGenerator.save_plot(fig, os.path.join(plot_dir, "R2_vs_MC_sweep.png"))

        fig, ax = PlotGenerator.setup_plot(
            "MC Sweep", "<Rg2> (A^2)", f"Rg2 vs MC Sweep {title_suffix}", loglog=False)
        if sweep_data:
            sweeps = [d[0] for d in sweep_data]
            Rg2s = [d[3] for d in sweep_data]
            ax.plot(sweeps, Rg2s, '-', color='C1', linewidth=1)
        PlotGenerator.save_plot(fig, os.path.join(plot_dir, "Rg2_vs_MC_sweep.png"))

        fig, ax = PlotGenerator.setup_plot(
            "Lag (sweeps)", "g1(t) (A^2)",
            f"Middle-Segment MSD {title_suffix}")
        g1 = results['g1']
        if g1:
            lags = sorted(g1.keys())
            vals = [g1[l] for l in lags]
            ax.plot(lags, vals, 'o-', color='C0', markersize=3)
            if len(lags) >= 2:
                slope, _, r2 = PlotGenerator.power_law_fit(lags, vals)
                ax.text(0.05, 0.95, f'slope = {slope:.3f} (R2={r2:.3f})',
                        transform=ax.transAxes, fontsize=11,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        PlotGenerator.save_plot(fig, os.path.join(plot_dir, "g1_middle_segment_msd.png"))

        fig, ax = PlotGenerator.setup_plot(
            "Lag (sweeps)", "g_CM(t) (A^2)",
            f"Center-of-Mass MSD {title_suffix}")
        gcm = results['gcm']
        if gcm:
            lags = sorted(gcm.keys())
            vals = [gcm[l] for l in lags]
            ax.plot(lags, vals, 'o-', color='C3', markersize=3)
            if len(lags) >= 2:
                slope, _, r2 = PlotGenerator.power_law_fit(lags, vals)
                ax.text(0.05, 0.95, f'slope = {slope:.3f} (R2={r2:.3f})',
                        transform=ax.transAxes, fontsize=11,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        PlotGenerator.save_plot(fig, os.path.join(plot_dir, "gcm_center_of_mass_msd.png"))

        fig, ax = PlotGenerator.setup_plot(
            "Lag (sweeps)", "g_R(t)",
            f"End-to-End Autocorrelation {title_suffix}")
        gr = results['gr']
        if gr:
            lags = sorted(gr.keys())
            vals = [gr[l] for l in lags]
            ax.plot(lags, vals, 'o-', color='C4', markersize=3)
            ax.axhline(y=1.0 / np.e, color='gray', linestyle='--', alpha=0.5,
                       label='1/e threshold')
            ax.legend(fontsize=10)
        PlotGenerator.save_plot(
            fig, os.path.join(plot_dir, "autocorrelation_end_to_end_vector.png"))

    @classmethod
    def plot_equilibration_per_phi(cls, phi_results: dict, base_dir: str, phi: float) -> None:
        phi_str = ConfigHelpers.format_phi(phi)
        eq_dir = os.path.join(base_dir, "04_equilibration_evidence", f"phi_{phi_str}")
        PlotGenerator.ensure_dir(eq_dir)

        for obs_idx, (obs_name, obs_label) in enumerate(
                [("R2", "<R2> (A^2)"), ("Rg2", "<Rg2> (A^2)")]):
            fig, ax = PlotGenerator.setup_plot(
                "MC Sweep", obs_label,
                f"{obs_name} vs MC Sweep (phi={phi_str}, All N)", loglog=False)
            for n in sorted(phi_results.keys()):
                data = phi_results[n]['sweep_data']
                if data:
                    sweeps = [d[0] for d in data]
                    vals = [d[2 + obs_idx] for d in data]
                    ax.plot(sweeps, vals, '-', linewidth=1, label=f'N={n}')
            ax.axvline(x=10000, color='black', linestyle='--', alpha=0.4,
                       linewidth=1, label='eq/prod boundary')
            ax.legend(fontsize=10)
            PlotGenerator.save_plot(
                fig, os.path.join(eq_dir, f"fig_{obs_name}_vs_sweep_all_N.png"))

    @classmethod
    def plot_R2_vs_N_per_phi(cls, all_results: dict, base_dir: str) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "N (chain length)", "<R2> (A^2)",
            "End-to-End Distance Squared vs N (per phi)")
        for phi in sorted(all_results.keys()):
            Ns = sorted(all_results[phi].keys())
            stats = [AnalysisMetrics.production_averaged_static_with_std(all_results[phi][n])
                     for n in Ns]
            mean_R2 = [s[0] for s in stats]
            std_R2 = [s[2] for s in stats]
            phi_str = ConfigHelpers.format_phi(phi)
            color = PlotGenerator.phi_color(phi)
            ax.errorbar(Ns, mean_R2, yerr=std_R2, fmt='o-', color=color,
                        markersize=6, capsize=4, label=f'phi={phi_str}')
            slope, intercept, r2 = PlotGenerator.power_law_fit(Ns, mean_R2)
            fit_y = np.exp(intercept) * np.array(Ns) ** slope
            ax.plot(Ns, fit_y, '--', color=color, alpha=0.5,
                    label=f'  2nu={slope:.3f} (R2={r2:.3f})')
        ax.legend(fontsize=8, ncol=2)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "01_static_properties", "fig_R2_vs_N_per_phi.png"))

    @classmethod
    def plot_Rg2_vs_N_per_phi(cls, all_results: dict, base_dir: str) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "N (chain length)", "<Rg2> (A^2)",
            "Radius of Gyration Squared vs N (per phi)")
        for phi in sorted(all_results.keys()):
            Ns = sorted(all_results[phi].keys())
            stats = [AnalysisMetrics.production_averaged_static_with_std(all_results[phi][n])
                     for n in Ns]
            mean_Rg2 = [s[1] for s in stats]
            std_Rg2 = [s[3] for s in stats]
            phi_str = ConfigHelpers.format_phi(phi)
            color = PlotGenerator.phi_color(phi)
            ax.errorbar(Ns, mean_Rg2, yerr=std_Rg2, fmt='s-', color=color,
                        markersize=6, capsize=4, label=f'phi={phi_str}')
            slope, intercept, r2 = PlotGenerator.power_law_fit(Ns, mean_Rg2)
            fit_y = np.exp(intercept) * np.array(Ns) ** slope
            ax.plot(Ns, fit_y, '--', color=color, alpha=0.5,
                    label=f'  2nu={slope:.3f} (R2={r2:.3f})')
        ax.legend(fontsize=8, ncol=2)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "01_static_properties", "fig_Rg2_vs_N_per_phi.png"))

    @classmethod
    def plot_2nu_vs_phi(cls, all_results: dict, base_dir: str) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "phi (volume fraction)", "2nu (scaling exponent)",
            "Static Scaling Exponent vs Density", loglog=False)
        ax.set_xscale('log')

        phis = sorted(all_results.keys())
        nu_R2 = []
        nu_Rg2 = []
        for phi in phis:
            Ns = sorted(all_results[phi].keys())
            mean_R2 = [AnalysisMetrics.production_averaged_static(all_results[phi][n])[0]
                       for n in Ns]
            mean_Rg2 = [AnalysisMetrics.production_averaged_static(all_results[phi][n])[1]
                        for n in Ns]
            s_R2, _, _ = PlotGenerator.power_law_fit(Ns, mean_R2)
            s_Rg2, _, _ = PlotGenerator.power_law_fit(Ns, mean_Rg2)
            nu_R2.append(s_R2)
            nu_Rg2.append(s_Rg2)

        ax.plot(phis, nu_R2, 'o-', color='C0', markersize=8, label='2nu from R2')
        ax.plot(phis, nu_Rg2, 's-', color='C1', markersize=8, label='2nu from Rg2')
        ax.axhline(y=1.20, color='red', linestyle='--', alpha=0.6,
                   label='Kuriata target (1.20)')
        ax.axhline(y=1.00, color='gray', linestyle=':', alpha=0.5,
                   label='Ideal chain (1.00)')
        ax.legend(fontsize=10)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "01_static_properties", "fig_2nu_vs_phi.png"))

    @classmethod
    def plot_ratio_R2_Rg2_vs_phi(cls, all_results: dict, base_dir: str,
                                 chain_lengths) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "phi (volume fraction)", "R2/Rg2",
            "R2/Rg2 Ratio vs Density (per N)", loglog=False)
        ax.set_xscale('log')
        phis = sorted(all_results.keys())

        for N in chain_lengths:
            ratio_vals = []
            for phi in phis:
                if N in all_results[phi]:
                    R2, Rg2 = AnalysisMetrics.production_averaged_static(all_results[phi][N])
                    ratio_vals.append(R2 / Rg2 if Rg2 > 0 else 0)
                else:
                    ratio_vals.append(0)
            ax.plot(phis, ratio_vals, 'o-', markersize=6, label=f'N={N}')

        ax.axhline(y=6.25, color='red', linestyle='--', alpha=0.6,
                   label='SAW expected (6.25)')
        ax.axhline(y=6.00, color='gray', linestyle=':', alpha=0.5,
                   label='Gaussian (6.00)')
        ax.legend(fontsize=9)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "01_static_properties", "fig_ratio_R2_Rg2_vs_phi.png"))

    @classmethod
    def plot_R2_Rg2_combined_dilute(cls, all_results: dict, base_dir: str) -> None:
        dilute_phi = min(all_results.keys())
        phi_str = ConfigHelpers.format_phi(dilute_phi)
        fig, ax = PlotGenerator.setup_plot(
            "N (chain length)", "Distance^2 (A^2)",
            f"R2 and Rg2 vs N (dilute, phi={phi_str})")
        phi_results = all_results[dilute_phi]
        Ns = sorted(phi_results.keys())
        mean_R2 = [AnalysisMetrics.production_averaged_static(phi_results[n])[0] for n in Ns]
        mean_Rg2 = [AnalysisMetrics.production_averaged_static(phi_results[n])[1] for n in Ns]

        ax.plot(Ns, mean_R2, 'o-', color='C0', markersize=8, label='<R2>')
        ax.plot(Ns, mean_Rg2, 's-', color='C1', markersize=8, label='<Rg2>')

        slope_R2, int_R2, r2_R2 = PlotGenerator.power_law_fit(Ns, mean_R2)
        slope_Rg2, int_Rg2, r2_Rg2 = PlotGenerator.power_law_fit(Ns, mean_Rg2)
        ax.plot(Ns, np.exp(int_R2) * np.array(Ns) ** slope_R2, '--', color='C0',
                alpha=0.5, label=f'R2 fit: 2nu={slope_R2:.3f} (R2={r2_R2:.3f})')
        ax.plot(Ns, np.exp(int_Rg2) * np.array(Ns) ** slope_Rg2, '--', color='C1',
                alpha=0.5, label=f'Rg2 fit: 2nu={slope_Rg2:.3f} (R2={r2_Rg2:.3f})')
        ax.axhline(y=0, visible=False)
        ax.legend(fontsize=10)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "01_static_properties", "fig_R2_Rg2_combined_dilute.png"))

    @classmethod
    def plot_g1_vs_sweep_per_phi(cls, all_results: dict, base_dir: str, ref_N: int = 100) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "Lag (sweeps)", "g1(t) (A^2)",
            f"Middle-Segment MSD vs Lag (N={ref_N}, per phi)")
        for phi in sorted(all_results.keys()):
            if ref_N in all_results[phi]:
                g1 = all_results[phi][ref_N]['g1']
                if g1:
                    lags = sorted(g1.keys())
                    vals = [g1[l] for l in lags]
                    ax.plot(lags, vals, 'o-', color=PlotGenerator.phi_color(phi),
                            markersize=3, label=f'phi={ConfigHelpers.format_phi(phi)}')
        ax.plot([], [], '--', color='gray', alpha=0.5, label='ref: t^0.5')
        ax.legend(fontsize=9)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "02_dynamic_properties", "fig_g1_vs_sweep_per_phi.png"))

    @classmethod
    def plot_gcm_vs_sweep_per_phi(cls, all_results: dict, base_dir: str, ref_N: int = 100) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "Lag (sweeps)", "g_CM(t) (A^2)",
            f"Center-of-Mass MSD vs Lag (N={ref_N}, per phi)")
        for phi in sorted(all_results.keys()):
            if ref_N in all_results[phi]:
                gcm = all_results[phi][ref_N]['gcm']
                if gcm:
                    lags = sorted(gcm.keys())
                    vals = [gcm[l] for l in lags]
                    ax.plot(lags, vals, 'o-', color=PlotGenerator.phi_color(phi),
                            markersize=3, label=f'phi={ConfigHelpers.format_phi(phi)}')
        ax.legend(fontsize=9)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "02_dynamic_properties", "fig_gcm_vs_sweep_per_phi.png"))

    @classmethod
    def plot_D_vs_N_per_phi(cls, all_results: dict, base_dir: str) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "N (chain length)", "D (A^2/sweep)",
            "Diffusion Coefficient vs N (per phi)")
        for phi in sorted(all_results.keys()):
            Ns = sorted(all_results[phi].keys())
            Ds = [AnalysisMetrics.estimate_diffusion_coefficient(all_results[phi][n]['gcm'])
                  for n in Ns]
            valid = [(n, d) for n, d in zip(Ns, Ds) if d > 0]
            if valid:
                vn, vd = zip(*valid)
                color = PlotGenerator.phi_color(phi)
                phi_str = ConfigHelpers.format_phi(phi)
                ax.plot(vn, vd, 'o-', color=color, markersize=6, label=f'phi={phi_str}')
                slope, intercept, r2 = PlotGenerator.power_law_fit(vn, vd)
                fit_y = np.exp(intercept) * np.array(vn) ** slope
                ax.plot(vn, fit_y, '--', color=color, alpha=0.5,
                        label=f'  exp={slope:.3f} (R2={r2:.3f})')
        ax.legend(fontsize=8, ncol=2)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "02_dynamic_properties", "fig_D_vs_N_per_phi.png"))

    @classmethod
    def plot_tauR_vs_N_per_phi(cls, all_results: dict, base_dir: str) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "N (chain length)", "tau_R (sweeps)",
            "Relaxation Time vs N (per phi)")
        for phi in sorted(all_results.keys()):
            Ns = sorted(all_results[phi].keys())
            taus = [AnalysisMetrics.estimate_relaxation_time(all_results[phi][n]['gr'])
                    for n in Ns]
            valid = [(n, t) for n, t in zip(Ns, taus) if t > 0]
            if valid:
                vn, vt = zip(*valid)
                color = PlotGenerator.phi_color(phi)
                phi_str = ConfigHelpers.format_phi(phi)
                ax.plot(vn, vt, 'o-', color=color, markersize=6, label=f'phi={phi_str}')
                slope, intercept, r2 = PlotGenerator.power_law_fit(vn, vt)
                fit_y = np.exp(intercept) * np.array(vn) ** slope
                ax.plot(vn, fit_y, '--', color=color, alpha=0.5,
                        label=f'  exp={slope:.3f} (R2={r2:.3f})')
        ax.legend(fontsize=8, ncol=2)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "02_dynamic_properties", "fig_tauR_vs_N_per_phi.png"))

    @classmethod
    def plot_D_exponent_vs_phi(cls, all_results: dict, base_dir: str) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "phi (volume fraction)", "D exponent (D ~ N^alpha)",
            "Diffusion Exponent vs Density", loglog=False)
        ax.set_xscale('log')
        phis = sorted(all_results.keys())
        exponents = []
        for phi in phis:
            Ns = sorted(all_results[phi].keys())
            Ds = [AnalysisMetrics.estimate_diffusion_coefficient(all_results[phi][n]['gcm'])
                  for n in Ns]
            valid = [(n, d) for n, d in zip(Ns, Ds) if d > 0]
            if valid:
                slope, _, _ = PlotGenerator.power_law_fit(
                    [v[0] for v in valid], [v[1] for v in valid])
                exponents.append(slope)
            else:
                exponents.append(0)
        ax.plot(phis, exponents, 'o-', color='C3', markersize=8, label='D exponent')
        ax.axhline(y=-1.00, color='red', linestyle='--', alpha=0.6,
                   label='Rouse theory (-1.00)')
        ax.legend(fontsize=10)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "02_dynamic_properties", "fig_D_exponent_vs_phi.png"))

    @classmethod
    def plot_tauR_exponent_vs_phi(cls, all_results: dict, base_dir: str) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "phi (volume fraction)", "tau_R exponent (tau_R ~ N^beta)",
            "Relaxation Time Exponent vs Density", loglog=False)
        ax.set_xscale('log')
        phis = sorted(all_results.keys())
        exponents = []
        for phi in phis:
            Ns = sorted(all_results[phi].keys())
            taus = [AnalysisMetrics.estimate_relaxation_time(all_results[phi][n]['gr'])
                    for n in Ns]
            valid = [(n, t) for n, t in zip(Ns, taus) if t > 0]
            if valid:
                slope, _, _ = PlotGenerator.power_law_fit(
                    [v[0] for v in valid], [v[1] for v in valid])
                exponents.append(slope)
            else:
                exponents.append(0)
        ax.plot(phis, exponents, 'o-', color='C4', markersize=8, label='tau_R exponent')
        ax.axhline(y=2.18, color='red', linestyle='--', alpha=0.6,
                   label='SAW theory (2.18)')
        ax.legend(fontsize=10)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "02_dynamic_properties", "fig_tauR_exponent_vs_phi.png"))

    @classmethod
    def plot_g1_shorttime_exponent_vs_phi(cls, all_results: dict, base_dir: str) -> None:
        fig, ax = PlotGenerator.setup_plot(
            "phi (volume fraction)", "g1 short-time exponent",
            "g1 Short-Time Exponent vs Density", loglog=False)
        ax.set_xscale('log')
        phis = sorted(all_results.keys())
        exponents = []
        for phi in phis:
            Ns = sorted(all_results[phi].keys())
            g1_exps = [AnalysisMetrics.estimate_g1_short_time_exponent(all_results[phi][n]['g1'])
                       for n in Ns]
            valid = [e for e in g1_exps if e > 0]
            exponents.append(float(np.mean(valid)) if valid else 0)
        ax.plot(phis, exponents, 'o-', color='C5', markersize=8, label='g1 short-time exp')
        ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.6,
                   label='Rouse theory (0.50)')
        ax.axhline(y=0.60, color='orange', linestyle=':', alpha=0.5,
                   label='Kuriata observed (0.60)')
        ax.legend(fontsize=10)
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "02_dynamic_properties",
                              "fig_g1_shorttime_exponent_vs_phi.png"))

    @classmethod
    def generate_all_plots(cls, all_results: dict, base_dir: str, chain_lengths) -> None:
        phis = sorted(all_results.keys())

        for phi in phis:
            for N in sorted(all_results[phi].keys()):
                cls.plot_per_state_point(all_results[phi][N], base_dir, N, phi)

        for phi in phis:
            cls.plot_equilibration_per_phi(all_results[phi], base_dir, phi)

        cls.plot_R2_vs_N_per_phi(all_results, base_dir)
        cls.plot_Rg2_vs_N_per_phi(all_results, base_dir)
        cls.plot_2nu_vs_phi(all_results, base_dir)
        cls.plot_ratio_R2_Rg2_vs_phi(all_results, base_dir, chain_lengths)
        cls.plot_R2_Rg2_combined_dilute(all_results, base_dir)

        cls.plot_g1_vs_sweep_per_phi(all_results, base_dir)
        cls.plot_gcm_vs_sweep_per_phi(all_results, base_dir)
        cls.plot_D_vs_N_per_phi(all_results, base_dir)
        cls.plot_tauR_vs_N_per_phi(all_results, base_dir)
        cls.plot_D_exponent_vs_phi(all_results, base_dir)
        cls.plot_tauR_exponent_vs_phi(all_results, base_dir)
        cls.plot_g1_shorttime_exponent_vs_phi(all_results, base_dir)
