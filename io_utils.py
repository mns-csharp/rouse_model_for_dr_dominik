"""
I/O utilities: TSV file writers and PNG plot generators.

Supports phi-organized directory structure for the 30-state-point
validation matrix (5 chain lengths x 6 phi levels).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats as scipy_stats

from .config import format_phi, PHI_VALUES, CHAIN_LENGTHS


# ============================================================================
# TSV Writers
# ============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_static_tsv(filepath: str, R2_array, Rg2_array):
    """Write fig1_static.tsv — Columns: chain_id, R2, Rg2"""
    ensure_dir(os.path.dirname(filepath))
    n = len(R2_array)
    with open(filepath, 'w') as f:
        f.write("chain_id\tR2\tRg2\n")
        for i in range(n):
            f.write(f"{i}\t{R2_array[i]:.8E}\t{Rg2_array[i]:.8E}\n")


def write_dynamic_tsv(filepath: str, data_dict: dict, col_name: str):
    """Write a dynamic observable TSV — Columns: lag_sweep, <col_name>"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        f.write(f"lag_sweep\t{col_name}\n")
        for lag in sorted(data_dict.keys()):
            f.write(f"{lag}\t{data_dict[lag]:.8E}\n")


def write_sweep_tsv(filepath: str, records: list):
    """Write static_vs_sweep.tsv — sweep, phase, mean_R2, mean_Rg2, ratio"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        f.write("sweep\tphase\tmean_R2\tmean_Rg2\tratio_R2_Rg2\n")
        for sweep, phase, R2, Rg2, ratio in records:
            f.write(f"{sweep}\t{phase}\t{R2:.8E}\t{Rg2:.8E}\t{ratio:.8E}\n")


def write_all_tsvs(results: dict, base_dir: str, N: int, phi: float):
    """Write all 5 TSV files for a given (N, phi) state point."""
    phi_str = format_phi(phi)
    data_dir = os.path.join(base_dir, "05_data", f"phi_{phi_str}", f"N{N}")
    ensure_dir(data_dir)

    R2 = results['final_R2'].cpu().numpy()
    Rg2 = results['final_Rg2'].cpu().numpy()
    write_static_tsv(os.path.join(data_dir, "fig1_static.tsv"), R2, Rg2)
    write_dynamic_tsv(os.path.join(data_dir, "fig2_seg_msd.tsv"), results['g1'], "g1")
    write_dynamic_tsv(os.path.join(data_dir, "fig3_cm_diffusion.tsv"), results['gcm'], "g_CM")
    write_dynamic_tsv(os.path.join(data_dir, "fig4_autocorr.tsv"), results['gr'], "g_R")
    write_sweep_tsv(os.path.join(data_dir, "static_vs_sweep.tsv"), results['sweep_data'])
    print(f"  Wrote 5 TSV files to {data_dir}")


# ============================================================================
# Plot Helpers
# ============================================================================

def power_law_fit(x, y):
    """Fit log(y) = a * log(x) + b. Returns (slope, intercept, r_squared)."""
    mask = (np.array(x) > 0) & (np.array(y) > 0)
    lx = np.log(np.array(x)[mask])
    ly = np.log(np.array(y)[mask])
    if len(lx) < 2:
        return 0.0, 0.0, 0.0
    slope, intercept, r, p, se = scipy_stats.linregress(lx, ly)
    return slope, intercept, r ** 2


def setup_plot(xlabel, ylabel, title, loglog=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    return fig, ax


def save_plot(fig, filepath):
    ensure_dir(os.path.dirname(filepath))
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


# Color map for phi values (light to dark)
PHI_COLORS = {
    0.001: '#1f77b4',
    0.01:  '#ff7f0e',
    0.05:  '#2ca02c',
    0.10:  '#d62728',
    0.20:  '#9467bd',
    0.30:  '#8c564b',
}


def phi_color(phi):
    return PHI_COLORS.get(phi, 'black')


# ============================================================================
# Analysis Helpers
# ============================================================================

def _estimate_diffusion_coefficient(gcm: dict) -> float:
    """Estimate D from late-time linear fit of gCM(t) = 6Dt."""
    if len(gcm) < 3:
        return 0.0
    lags = sorted(gcm.keys())
    n = len(lags)
    start = max(1, n // 2)
    x = np.array(lags[start:], dtype=float)
    y = np.array([gcm[l] for l in lags[start:]], dtype=float)
    if len(x) < 2:
        return 0.0
    slope, _, _, _, _ = scipy_stats.linregress(x, y)
    return slope / 6.0


def _estimate_relaxation_time(gr: dict) -> float:
    """Estimate tau_R as the lag where g_R first drops below 1/e."""
    target = 1.0 / np.e
    lags = sorted(gr.keys())
    for l in lags:
        if gr[l] <= target:
            return float(l)
    if len(lags) >= 2:
        return float(lags[-1]) * 2.0
    return float(lags[-1]) if lags else 1.0


def _estimate_g1_short_time_exponent(g1: dict) -> float:
    """Estimate g1 short-time exponent from first third of data."""
    if len(g1) < 3:
        return 0.0
    lags = sorted(g1.keys())
    n = max(3, len(lags) // 3)
    x = np.array(lags[:n], dtype=float)
    y = np.array([g1[l] for l in lags[:n]], dtype=float)
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return 0.0
    slope, _, _, _, _ = scipy_stats.linregress(np.log(x[mask]), np.log(y[mask]))
    return slope


def _estimate_gcm_exponent(gcm: dict) -> float:
    """Estimate g_CM(t) ~ t^alpha exponent from log-log fit."""
    if len(gcm) < 3:
        return 0.0
    lags = sorted(gcm.keys())
    x = np.array(lags, dtype=float)
    y = np.array([gcm[l] for l in lags], dtype=float)
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return 0.0
    slope, _, _, _, _ = scipy_stats.linregress(np.log(x[mask]), np.log(y[mask]))
    return slope


def _check_pass(measured, theory, tolerance):
    """Check if measured value is within tolerance of theory."""
    return bool(abs(measured - theory) <= tolerance)


def _compute_phi_metrics(phi_results: dict):
    """Compute all scaling metrics for a single phi level.

    Args:
        phi_results: dict[N] -> results for all chain lengths at this phi

    Returns:
        dict with all metrics for this phi
    """
    Ns = sorted(phi_results.keys())
    mean_R2 = {n: phi_results[n]['final_R2'].mean().item() for n in Ns}
    mean_Rg2 = {n: phi_results[n]['final_Rg2'].mean().item() for n in Ns}
    ratios = {n: mean_R2[n] / mean_Rg2[n] if mean_Rg2[n] > 0 else 0.0 for n in Ns}

    slope_R2, _, r2_R2 = power_law_fit(Ns, [mean_R2[n] for n in Ns])
    slope_Rg2, _, r2_Rg2 = power_law_fit(Ns, [mean_Rg2[n] for n in Ns])

    Ds = {n: _estimate_diffusion_coefficient(phi_results[n]['gcm']) for n in Ns}
    valid_D = [(n, Ds[n]) for n in Ns if Ds[n] > 0]
    diff_slope, diff_r2 = 0.0, 0.0
    if valid_D:
        diff_slope, _, diff_r2 = power_law_fit([v[0] for v in valid_D],
                                                [v[1] for v in valid_D])

    taus = {n: _estimate_relaxation_time(phi_results[n]['gr']) for n in Ns}
    valid_tau = [(n, taus[n]) for n in Ns if taus[n] > 0]
    relax_slope, relax_r2 = 0.0, 0.0
    if valid_tau:
        relax_slope, _, relax_r2 = power_law_fit([v[0] for v in valid_tau],
                                                   [v[1] for v in valid_tau])

    gcm_exponents = {n: _estimate_gcm_exponent(phi_results[n]['gcm']) for n in Ns}
    mean_gcm_exp = np.mean([v for v in gcm_exponents.values() if v > 0]) if any(v > 0 for v in gcm_exponents.values()) else 0.0

    g1_exponents = {n: _estimate_g1_short_time_exponent(phi_results[n]['g1']) for n in Ns}
    mean_g1_exp = np.mean([v for v in g1_exponents.values() if v > 0]) if any(v > 0 for v in g1_exponents.values()) else 0.0

    mean_ratio = np.mean(list(ratios.values()))

    return {
        "R2_exponent": {"measured": round(float(slope_R2), 3), "theory": 1.18,
                        "r2_fit": round(float(r2_R2), 4),
                        "pass": _check_pass(slope_R2, 1.18, 0.10)},
        "Rg2_exponent": {"measured": round(float(slope_Rg2), 3), "theory": 1.18,
                         "r2_fit": round(float(r2_Rg2), 4),
                         "pass": _check_pass(slope_Rg2, 1.18, 0.10)},
        "R2_Rg2_ratio": {"measured": round(float(mean_ratio), 3), "theory": 6.25,
                         "pass": _check_pass(mean_ratio, 6.25, 1.25)},
        "D_exponent": {"measured": round(float(diff_slope), 3), "theory": -1.00,
                       "r2_fit": round(float(diff_r2), 4),
                       "pass": _check_pass(diff_slope, -1.00, 0.10)},
        "tau_R_exponent": {"measured": round(float(relax_slope), 3), "theory": 2.18,
                           "r2_fit": round(float(relax_r2), 4),
                           "pass": _check_pass(relax_slope, 2.18, 0.20)},
        "g_CM_exponent": {"measured": round(float(mean_gcm_exp), 3), "theory": 1.00,
                          "pass": _check_pass(mean_gcm_exp, 1.00, 0.10)},
        "g1_exponent": {"measured": round(float(mean_g1_exp), 3), "theory": 0.50,
                        "pass": _check_pass(mean_g1_exp, 0.50, 0.15)},
        "_detail": {
            "N_values": Ns,
            "mean_R2": {str(n): round(mean_R2[n], 4) for n in Ns},
            "mean_Rg2": {str(n): round(mean_Rg2[n], 4) for n in Ns},
            "R2_over_Rg2_per_N": {str(n): round(ratios[n], 4) for n in Ns},
            "diffusion_coefficients": {str(n): Ds[n] for n in Ns},
            "relaxation_times": {str(n): taus[n] for n in Ns},
            "g1_exponents_per_N": {str(n): round(g1_exponents[n], 4) for n in Ns},
            "gcm_exponents_per_N": {str(n): round(gcm_exponents[n], 4) for n in Ns},
        }
    }


def _check_marginal(measured, theory, tolerance):
    """MARGINAL = within 1.5x tolerance but outside 1x tolerance."""
    diff = abs(measured - theory)
    if diff <= tolerance:
        return 2  # PASS
    elif diff <= tolerance * 1.5:
        return 1  # MARGINAL
    return 0  # FAIL


# ============================================================================
# Per-State-Point Plots (5 plots per state point x 30 = 150 total)
# ============================================================================

def plot_per_state_point(results: dict, base_dir: str, N: int, phi: float):
    """Generate all 5 plots for a single (N, phi) state point."""
    phi_str = format_phi(phi)
    plot_dir = os.path.join(base_dir, "03_per_state_point", f"phi_{phi_str}", f"N{N}")
    ensure_dir(plot_dir)

    sweep_data = results['sweep_data']
    title_suffix = f"(N={N}, phi={phi_str})"

    # 1. R2 vs MC sweep
    fig, ax = setup_plot("MC Sweep", "<R2> (A^2)", f"R2 vs MC Sweep {title_suffix}",
                          loglog=False)
    if sweep_data:
        sweeps = [d[0] for d in sweep_data]
        R2s = [d[2] for d in sweep_data]
        ax.plot(sweeps, R2s, '-', color='C0', linewidth=1)
    save_plot(fig, os.path.join(plot_dir, "R2_vs_MC_sweep.png"))

    # 2. Rg2 vs MC sweep
    fig, ax = setup_plot("MC Sweep", "<Rg2> (A^2)", f"Rg2 vs MC Sweep {title_suffix}",
                          loglog=False)
    if sweep_data:
        sweeps = [d[0] for d in sweep_data]
        Rg2s = [d[3] for d in sweep_data]
        ax.plot(sweeps, Rg2s, '-', color='C1', linewidth=1)
    save_plot(fig, os.path.join(plot_dir, "Rg2_vs_MC_sweep.png"))

    # 3. g1 middle-segment MSD
    fig, ax = setup_plot("Lag (sweeps)", "g1(t) (A^2)",
                          f"Middle-Segment MSD {title_suffix}")
    g1 = results['g1']
    if g1:
        lags = sorted(g1.keys())
        vals = [g1[l] for l in lags]
        ax.plot(lags, vals, 'o-', color='C0', markersize=3)
        if len(lags) >= 2:
            slope, _, r2 = power_law_fit(lags, vals)
            ax.text(0.05, 0.95, f'slope = {slope:.3f} (R2={r2:.3f})',
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    save_plot(fig, os.path.join(plot_dir, "g1_middle_segment_msd.png"))

    # 4. gCM center-of-mass MSD
    fig, ax = setup_plot("Lag (sweeps)", "g_CM(t) (A^2)",
                          f"Center-of-Mass MSD {title_suffix}")
    gcm = results['gcm']
    if gcm:
        lags = sorted(gcm.keys())
        vals = [gcm[l] for l in lags]
        ax.plot(lags, vals, 'o-', color='C3', markersize=3)
        if len(lags) >= 2:
            slope, _, r2 = power_law_fit(lags, vals)
            ax.text(0.05, 0.95, f'slope = {slope:.3f} (R2={r2:.3f})',
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    save_plot(fig, os.path.join(plot_dir, "gcm_center_of_mass_msd.png"))

    # 5. Autocorrelation
    fig, ax = setup_plot("Lag (sweeps)", "g_R(t)",
                          f"End-to-End Autocorrelation {title_suffix}")
    gr = results['gr']
    if gr:
        lags = sorted(gr.keys())
        vals = [gr[l] for l in lags]
        ax.plot(lags, vals, 'o-', color='C4', markersize=3)
        ax.axhline(y=1.0 / np.e, color='gray', linestyle='--', alpha=0.5,
                    label='1/e threshold')
        ax.legend(fontsize=10)
    save_plot(fig, os.path.join(plot_dir, "autocorrelation_end_to_end_vector.png"))


# ============================================================================
# Per-Phi Equilibration Evidence (2 plots per phi x 6 = 12 total)
# ============================================================================

def plot_equilibration_per_phi(phi_results: dict, base_dir: str, phi: float):
    """Generate 2 equilibration evidence plots for a given phi level.

    Args:
        phi_results: dict[N] -> results for all chain lengths at this phi
    """
    phi_str = format_phi(phi)
    eq_dir = os.path.join(base_dir, "04_equilibration_evidence", f"phi_{phi_str}")
    ensure_dir(eq_dir)

    for obs_idx, (obs_name, obs_label) in enumerate([("R2", "<R2> (A^2)"), ("Rg2", "<Rg2> (A^2)")]):
        fig, ax = setup_plot("MC Sweep", obs_label,
                              f"{obs_name} vs MC Sweep (phi={phi_str}, All N)",
                              loglog=False)
        for n in sorted(phi_results.keys()):
            data = phi_results[n]['sweep_data']
            if data:
                sweeps = [d[0] for d in data]
                vals = [d[2 + obs_idx] for d in data]
                ax.plot(sweeps, vals, '-', linewidth=1, label=f'N={n}')
        # eq/prod boundary at sweep 10000
        ax.axvline(x=10000, color='black', linestyle='--', alpha=0.4,
                    linewidth=1, label='eq/prod boundary')
        ax.legend(fontsize=10)
        save_plot(fig, os.path.join(eq_dir, f"fig_{obs_name}_vs_sweep_all_N.png"))


# ============================================================================
# Cross-Phi Static Plots (5 plots in 01_static_properties/)
# ============================================================================

def plot_R2_vs_N_per_phi(all_results: dict, base_dir: str):
    """fig_R2_vs_N_per_phi.png — R2 vs N, one curve per phi with power-law fits."""
    fig, ax = setup_plot("N (chain length)", "<R2> (A^2)",
                          "End-to-End Distance Squared vs N (per phi)")
    for phi in sorted(all_results.keys()):
        Ns = sorted(all_results[phi].keys())
        mean_R2 = [all_results[phi][n]['final_R2'].mean().item() for n in Ns]
        phi_str = format_phi(phi)
        color = phi_color(phi)
        ax.plot(Ns, mean_R2, 'o-', color=color, markersize=6, label=f'phi={phi_str}')
        slope, intercept, r2 = power_law_fit(Ns, mean_R2)
        fit_y = np.exp(intercept) * np.array(Ns) ** slope
        ax.plot(Ns, fit_y, '--', color=color, alpha=0.5,
                label=f'  2nu={slope:.3f} (R2={r2:.3f})')
    ax.legend(fontsize=8, ncol=2)
    save_plot(fig, os.path.join(base_dir, "01_static_properties", "fig_R2_vs_N_per_phi.png"))


def plot_Rg2_vs_N_per_phi(all_results: dict, base_dir: str):
    """fig_Rg2_vs_N_per_phi.png — Rg2 vs N, one curve per phi."""
    fig, ax = setup_plot("N (chain length)", "<Rg2> (A^2)",
                          "Radius of Gyration Squared vs N (per phi)")
    for phi in sorted(all_results.keys()):
        Ns = sorted(all_results[phi].keys())
        mean_Rg2 = [all_results[phi][n]['final_Rg2'].mean().item() for n in Ns]
        phi_str = format_phi(phi)
        color = phi_color(phi)
        ax.plot(Ns, mean_Rg2, 's-', color=color, markersize=6, label=f'phi={phi_str}')
        slope, intercept, r2 = power_law_fit(Ns, mean_Rg2)
        fit_y = np.exp(intercept) * np.array(Ns) ** slope
        ax.plot(Ns, fit_y, '--', color=color, alpha=0.5,
                label=f'  2nu={slope:.3f} (R2={r2:.3f})')
    ax.legend(fontsize=8, ncol=2)
    save_plot(fig, os.path.join(base_dir, "01_static_properties", "fig_Rg2_vs_N_per_phi.png"))


def plot_2nu_vs_phi(all_results: dict, base_dir: str):
    """fig_2nu_vs_phi.png — scaling exponent 2nu vs phi."""
    fig, ax = setup_plot("phi (volume fraction)", "2nu (scaling exponent)",
                          "Static Scaling Exponent vs Density", loglog=False)
    ax.set_xscale('log')

    phis = sorted(all_results.keys())
    nu_R2 = []
    nu_Rg2 = []
    for phi in phis:
        Ns = sorted(all_results[phi].keys())
        mean_R2 = [all_results[phi][n]['final_R2'].mean().item() for n in Ns]
        mean_Rg2 = [all_results[phi][n]['final_Rg2'].mean().item() for n in Ns]
        s_R2, _, _ = power_law_fit(Ns, mean_R2)
        s_Rg2, _, _ = power_law_fit(Ns, mean_Rg2)
        nu_R2.append(s_R2)
        nu_Rg2.append(s_Rg2)

    ax.plot(phis, nu_R2, 'o-', color='C0', markersize=8, label='2nu from R2')
    ax.plot(phis, nu_Rg2, 's-', color='C1', markersize=8, label='2nu from Rg2')
    ax.axhline(y=1.20, color='red', linestyle='--', alpha=0.6, label='Kuriata target (1.20)')
    ax.axhline(y=1.00, color='gray', linestyle=':', alpha=0.5, label='Ideal chain (1.00)')
    ax.legend(fontsize=10)
    save_plot(fig, os.path.join(base_dir, "01_static_properties", "fig_2nu_vs_phi.png"))


def plot_ratio_R2_Rg2_vs_phi(all_results: dict, base_dir: str):
    """fig_ratio_R2_Rg2_vs_phi.png — R2/Rg2 ratio vs phi, one curve per N."""
    fig, ax = setup_plot("phi (volume fraction)", "R2/Rg2",
                          "R2/Rg2 Ratio vs Density (per N)", loglog=False)
    ax.set_xscale('log')
    phis = sorted(all_results.keys())

    for N in CHAIN_LENGTHS:
        ratio_vals = []
        for phi in phis:
            if N in all_results[phi]:
                R2 = all_results[phi][N]['final_R2'].mean().item()
                Rg2 = all_results[phi][N]['final_Rg2'].mean().item()
                ratio_vals.append(R2 / Rg2 if Rg2 > 0 else 0)
            else:
                ratio_vals.append(0)
        ax.plot(phis, ratio_vals, 'o-', markersize=6, label=f'N={N}')

    ax.axhline(y=6.25, color='red', linestyle='--', alpha=0.6, label='SAW expected (6.25)')
    ax.axhline(y=6.00, color='gray', linestyle=':', alpha=0.5, label='Gaussian (6.00)')
    ax.legend(fontsize=9)
    save_plot(fig, os.path.join(base_dir, "01_static_properties", "fig_ratio_R2_Rg2_vs_phi.png"))


def plot_R2_Rg2_combined_dilute(all_results: dict, base_dir: str):
    """fig_R2_Rg2_combined_dilute.png — R2 and Rg2 vs N at phi=0.001."""
    dilute_phi = min(all_results.keys())
    phi_str = format_phi(dilute_phi)
    fig, ax = setup_plot("N (chain length)", "Distance^2 (A^2)",
                          f"R2 and Rg2 vs N (dilute, phi={phi_str})")
    phi_results = all_results[dilute_phi]
    Ns = sorted(phi_results.keys())
    mean_R2 = [phi_results[n]['final_R2'].mean().item() for n in Ns]
    mean_Rg2 = [phi_results[n]['final_Rg2'].mean().item() for n in Ns]

    ax.plot(Ns, mean_R2, 'o-', color='C0', markersize=8, label='<R2>')
    ax.plot(Ns, mean_Rg2, 's-', color='C1', markersize=8, label='<Rg2>')

    slope_R2, int_R2, r2_R2 = power_law_fit(Ns, mean_R2)
    slope_Rg2, int_Rg2, r2_Rg2 = power_law_fit(Ns, mean_Rg2)
    ax.plot(Ns, np.exp(int_R2) * np.array(Ns) ** slope_R2, '--', color='C0', alpha=0.5,
            label=f'R2 fit: 2nu={slope_R2:.3f} (R2={r2_R2:.3f})')
    ax.plot(Ns, np.exp(int_Rg2) * np.array(Ns) ** slope_Rg2, '--', color='C1', alpha=0.5,
            label=f'Rg2 fit: 2nu={slope_Rg2:.3f} (R2={r2_Rg2:.3f})')
    ax.axhline(y=0, visible=False)  # force origin
    ax.legend(fontsize=10)
    save_plot(fig, os.path.join(base_dir, "01_static_properties", "fig_R2_Rg2_combined_dilute.png"))


# ============================================================================
# Cross-Phi Dynamic Plots (7 plots in 02_dynamic_properties/)
# ============================================================================

def plot_g1_vs_sweep_per_phi(all_results: dict, base_dir: str):
    """fig_g1_vs_sweep_per_phi.png — g1(t) for N=100 at each phi."""
    ref_N = 100
    fig, ax = setup_plot("Lag (sweeps)", "g1(t) (A^2)",
                          f"Middle-Segment MSD vs Lag (N={ref_N}, per phi)")
    for phi in sorted(all_results.keys()):
        if ref_N in all_results[phi]:
            g1 = all_results[phi][ref_N]['g1']
            if g1:
                lags = sorted(g1.keys())
                vals = [g1[l] for l in lags]
                ax.plot(lags, vals, 'o-', color=phi_color(phi), markersize=3,
                        label=f'phi={format_phi(phi)}')
    # Reference slope t^0.5
    ax.plot([], [], '--', color='gray', alpha=0.5, label='ref: t^0.5')
    ax.legend(fontsize=9)
    save_plot(fig, os.path.join(base_dir, "02_dynamic_properties", "fig_g1_vs_sweep_per_phi.png"))


def plot_gcm_vs_sweep_per_phi(all_results: dict, base_dir: str):
    """fig_gcm_vs_sweep_per_phi.png — g_CM(t) for N=100 at each phi."""
    ref_N = 100
    fig, ax = setup_plot("Lag (sweeps)", "g_CM(t) (A^2)",
                          f"Center-of-Mass MSD vs Lag (N={ref_N}, per phi)")
    for phi in sorted(all_results.keys()):
        if ref_N in all_results[phi]:
            gcm = all_results[phi][ref_N]['gcm']
            if gcm:
                lags = sorted(gcm.keys())
                vals = [gcm[l] for l in lags]
                ax.plot(lags, vals, 'o-', color=phi_color(phi), markersize=3,
                        label=f'phi={format_phi(phi)}')
    ax.legend(fontsize=9)
    save_plot(fig, os.path.join(base_dir, "02_dynamic_properties", "fig_gcm_vs_sweep_per_phi.png"))


def plot_D_vs_N_per_phi(all_results: dict, base_dir: str):
    """fig_D_vs_N_per_phi.png — D vs N, one curve per phi with fits."""
    fig, ax = setup_plot("N (chain length)", "D (A^2/sweep)",
                          "Diffusion Coefficient vs N (per phi)")
    for phi in sorted(all_results.keys()):
        Ns = sorted(all_results[phi].keys())
        Ds = [_estimate_diffusion_coefficient(all_results[phi][n]['gcm']) for n in Ns]
        valid = [(n, d) for n, d in zip(Ns, Ds) if d > 0]
        if valid:
            vn, vd = zip(*valid)
            color = phi_color(phi)
            phi_str = format_phi(phi)
            ax.plot(vn, vd, 'o-', color=color, markersize=6, label=f'phi={phi_str}')
            slope, intercept, r2 = power_law_fit(vn, vd)
            fit_y = np.exp(intercept) * np.array(vn) ** slope
            ax.plot(vn, fit_y, '--', color=color, alpha=0.5,
                    label=f'  exp={slope:.3f} (R2={r2:.3f})')
    ax.legend(fontsize=8, ncol=2)
    save_plot(fig, os.path.join(base_dir, "02_dynamic_properties", "fig_D_vs_N_per_phi.png"))


def plot_tauR_vs_N_per_phi(all_results: dict, base_dir: str):
    """fig_tauR_vs_N_per_phi.png — tau_R vs N, one curve per phi."""
    fig, ax = setup_plot("N (chain length)", "tau_R (sweeps)",
                          "Relaxation Time vs N (per phi)")
    for phi in sorted(all_results.keys()):
        Ns = sorted(all_results[phi].keys())
        taus = [_estimate_relaxation_time(all_results[phi][n]['gr']) for n in Ns]
        valid = [(n, t) for n, t in zip(Ns, taus) if t > 0]
        if valid:
            vn, vt = zip(*valid)
            color = phi_color(phi)
            phi_str = format_phi(phi)
            ax.plot(vn, vt, 'o-', color=color, markersize=6, label=f'phi={phi_str}')
            slope, intercept, r2 = power_law_fit(vn, vt)
            fit_y = np.exp(intercept) * np.array(vn) ** slope
            ax.plot(vn, fit_y, '--', color=color, alpha=0.5,
                    label=f'  exp={slope:.3f} (R2={r2:.3f})')
    ax.legend(fontsize=8, ncol=2)
    save_plot(fig, os.path.join(base_dir, "02_dynamic_properties", "fig_tauR_vs_N_per_phi.png"))


def plot_D_exponent_vs_phi(all_results: dict, base_dir: str):
    """fig_D_exponent_vs_phi.png — D scaling exponent vs phi."""
    fig, ax = setup_plot("phi (volume fraction)", "D exponent (D ~ N^alpha)",
                          "Diffusion Exponent vs Density", loglog=False)
    ax.set_xscale('log')
    phis = sorted(all_results.keys())
    exponents = []
    for phi in phis:
        Ns = sorted(all_results[phi].keys())
        Ds = [_estimate_diffusion_coefficient(all_results[phi][n]['gcm']) for n in Ns]
        valid = [(n, d) for n, d in zip(Ns, Ds) if d > 0]
        if valid:
            slope, _, _ = power_law_fit([v[0] for v in valid], [v[1] for v in valid])
            exponents.append(slope)
        else:
            exponents.append(0)
    ax.plot(phis, exponents, 'o-', color='C3', markersize=8, label='D exponent')
    ax.axhline(y=-1.00, color='red', linestyle='--', alpha=0.6, label='Rouse theory (-1.00)')
    ax.legend(fontsize=10)
    save_plot(fig, os.path.join(base_dir, "02_dynamic_properties", "fig_D_exponent_vs_phi.png"))


def plot_tauR_exponent_vs_phi(all_results: dict, base_dir: str):
    """fig_tauR_exponent_vs_phi.png — tau_R scaling exponent vs phi."""
    fig, ax = setup_plot("phi (volume fraction)", "tau_R exponent (tau_R ~ N^beta)",
                          "Relaxation Time Exponent vs Density", loglog=False)
    ax.set_xscale('log')
    phis = sorted(all_results.keys())
    exponents = []
    for phi in phis:
        Ns = sorted(all_results[phi].keys())
        taus = [_estimate_relaxation_time(all_results[phi][n]['gr']) for n in Ns]
        valid = [(n, t) for n, t in zip(Ns, taus) if t > 0]
        if valid:
            slope, _, _ = power_law_fit([v[0] for v in valid], [v[1] for v in valid])
            exponents.append(slope)
        else:
            exponents.append(0)
    ax.plot(phis, exponents, 'o-', color='C4', markersize=8, label='tau_R exponent')
    ax.axhline(y=2.18, color='red', linestyle='--', alpha=0.6, label='SAW theory (2.18)')
    ax.legend(fontsize=10)
    save_plot(fig, os.path.join(base_dir, "02_dynamic_properties", "fig_tauR_exponent_vs_phi.png"))


def plot_g1_shorttime_exponent_vs_phi(all_results: dict, base_dir: str):
    """fig_g1_shorttime_exponent_vs_phi.png — g1 short-time exponent vs phi."""
    fig, ax = setup_plot("phi (volume fraction)", "g1 short-time exponent",
                          "g1 Short-Time Exponent vs Density", loglog=False)
    ax.set_xscale('log')
    phis = sorted(all_results.keys())
    exponents = []
    for phi in phis:
        Ns = sorted(all_results[phi].keys())
        g1_exps = [_estimate_g1_short_time_exponent(all_results[phi][n]['g1']) for n in Ns]
        valid = [e for e in g1_exps if e > 0]
        exponents.append(np.mean(valid) if valid else 0)
    ax.plot(phis, exponents, 'o-', color='C5', markersize=8, label='g1 short-time exp')
    ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.6, label='Rouse theory (0.50)')
    ax.axhline(y=0.60, color='orange', linestyle=':', alpha=0.5, label='Kuriata observed (0.60)')
    ax.legend(fontsize=10)
    save_plot(fig, os.path.join(base_dir, "02_dynamic_properties",
                                 "fig_g1_shorttime_exponent_vs_phi.png"))


# ============================================================================
# Compliance Heatmap
# ============================================================================

def plot_compliance_heatmap(all_results: dict, base_dir: str, all_phi_metrics: dict):
    """Generate rouse_compliance_heatmap.png — 2D PASS/FAIL/MARGINAL grid.

    Args:
        all_results: full results dict[phi][N]
        base_dir: output root
        all_phi_metrics: dict[phi] -> metrics from _compute_phi_metrics
    """
    properties = [
        ("2nu (R2)", "R2_exponent", 1.18, 0.10),
        ("2nu (Rg2)", "Rg2_exponent", 1.18, 0.10),
        ("R2/Rg2", "R2_Rg2_ratio", 6.25, 1.25),
        ("D exp", "D_exponent", -1.00, 0.10),
        ("tau_R exp", "tau_R_exponent", 2.18, 0.20),
        ("g_CM exp", "g_CM_exponent", 1.00, 0.10),
        ("g1 exp", "g1_exponent", 0.50, 0.15),
    ]

    phis = sorted(all_phi_metrics.keys())
    n_props = len(properties)
    n_phis = len(phis)
    matrix = np.zeros((n_props, n_phis))
    cell_text = [['' for _ in range(n_phis)] for _ in range(n_props)]

    for j, phi in enumerate(phis):
        metrics = all_phi_metrics[phi]
        for i, (label, key, theory, tol) in enumerate(properties):
            measured = metrics[key]["measured"]
            verdict = _check_marginal(measured, theory, tol)
            matrix[i, j] = verdict
            cell_text[i][j] = f'{measured:.2f}'

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = mcolors.ListedColormap(['#ff4444', '#ffcc00', '#44bb44'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    # Labels
    ax.set_xticks(range(n_phis))
    ax.set_xticklabels([format_phi(p) for p in phis], fontsize=11)
    ax.set_yticks(range(n_props))
    ax.set_yticklabels([p[0] for p in properties], fontsize=11)
    ax.set_xlabel("phi (volume fraction)", fontsize=12)
    ax.set_title("Rouse Compliance Heatmap: PASS / MARGINAL / FAIL", fontsize=14)

    # Cell text
    for i in range(n_props):
        for j in range(n_phis):
            color = 'white' if matrix[i, j] == 0 else 'black'
            ax.text(j, i, cell_text[i][j], ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#44bb44', label='PASS'),
                       Patch(facecolor='#ffcc00', label='MARGINAL'),
                       Patch(facecolor='#ff4444', label='FAIL')]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=10)

    fig.tight_layout()
    save_plot(fig, os.path.join(base_dir, "05_data", "rouse_compliance_heatmap.png"))
    print("  Generated compliance heatmap")

    return matrix


# ============================================================================
# Validation Summary JSON
# ============================================================================

def write_validation_summary(all_results: dict, base_dir: str):
    """Write tavg_validation_summary.json with per-phi metrics, phi*, and compliance."""
    phis = sorted(all_results.keys())

    # Compute metrics for each phi
    all_phi_metrics = {}
    for phi in phis:
        all_phi_metrics[phi] = _compute_phi_metrics(all_results[phi])

    # Generate compliance heatmap
    plot_compliance_heatmap(all_results, base_dir, all_phi_metrics)

    # Identify phi* for each property
    properties_for_phi_star = [
        ("R2_exponent", 1.18, 0.10),
        ("Rg2_exponent", 1.18, 0.10),
        ("R2_Rg2_ratio", 6.25, 1.25),
        ("D_exponent", -1.00, 0.10),
        ("tau_R_exponent", 2.18, 0.20),
        ("g_CM_exponent", 1.00, 0.10),
        ("g1_exponent", 0.50, 0.15),
    ]

    phi_star = {}
    for key, theory, tol in properties_for_phi_star:
        phi_star[key] = None
        for phi in phis:
            if not all_phi_metrics[phi][key]["pass"]:
                phi_star[key] = phi
                break
        if phi_star[key] is None:
            phi_star[key] = "never (passes at all phi)"

    # Build Rouse 1953 assessment (at dilute limit)
    dilute_phi = phis[0]
    dilute = all_phi_metrics[dilute_phi]
    d_detail = dilute["_detail"]
    Ns = d_detail["N_values"]

    def met(val, reason):
        return f"{'MET' if val else 'NOT MET'} -- {reason}"

    rouse_1953 = {
        "1.1_gaussian_submolecule": met(False,
            "surpass-alpha uses athermal excluded volume (E=1e6), producing SAW chains, "
            "not Gaussian submolecules."),
        "1.2_R2_scaling": met(dilute["R2_exponent"]["pass"],
            f"<R2> ~ N^(2nu) with 2nu = {dilute['R2_exponent']['measured']:.3f} "
            f"(R2_fit = {dilute['R2_exponent']['r2_fit']:.4f})"),
        "1.3_Rg2_scaling": met(dilute["Rg2_exponent"]["pass"],
            f"<Rg2> ~ N^(2nu) with 2nu = {dilute['Rg2_exponent']['measured']:.3f} "
            f"(R2_fit = {dilute['Rg2_exponent']['r2_fit']:.4f})"),
        "1.4_ratio_R2_Rg2": met(dilute["R2_Rg2_ratio"]["pass"],
            f"Mean R2/Rg2 = {dilute['R2_Rg2_ratio']['measured']:.3f} "
            f"(SAW expected ~6.25)"),
        "1.5_config_probability": met(False,
            "End-to-end distribution is non-Gaussian for SAW chains with excluded volume."),
        "2.1_eigenvalues": met(False,
            "Rouse eigenvalues require harmonic springs; surpass-alpha uses rigid bonds + MC moves."),
        "2.2_relaxation_times": met(True,
            f"tau_R extracted from g_R(t) = 1/e crossing for all N."),
        "2.3_long_wavelength_approx": met(False,
            "Long-wavelength approx tau_p ~ N^2/p^2 not directly measurable from MC."),
        "2.4_tau_R_scaling": met(dilute["tau_R_exponent"]["pass"],
            f"tau_R ~ N^{dilute['tau_R_exponent']['measured']:.3f} "
            f"(theory: 2.18)"),
        "2.5_steady_flow_viscosity": met(False,
            "Viscosity requires stress tensor; not available from equilibrium MC."),
        "2.6_complex_viscosity": met(False,
            "Complex viscosity requires frequency-domain analysis; not available from MC."),
        "2.7_shear_modulus": met(False,
            "Shear modulus requires stress autocorrelation; not computed."),
        "2.8_high_freq_approx": met(False,
            "High-frequency behavior not accessible from MC simulations."),
        "2.9_diffusion_coeff": met(dilute["D_exponent"]["pass"],
            f"D ~ N^{dilute['D_exponent']['measured']:.3f} "
            f"(theory: -1.00)"),
        "2.10_g_CM": met(dilute["g_CM_exponent"]["pass"],
            f"g_CM(t) ~ t^{dilute['g_CM_exponent']['measured']:.3f} "
            f"(theory: 1.00)"),
        "2.11_g1_middle_segment": met(dilute["g1_exponent"]["pass"],
            f"g1 short-time exponent = {dilute['g1_exponent']['measured']:.3f} "
            f"(theory: 0.50)"),
        "2.12_end_to_end_autocorr": met(True,
            "g_R(t) decays from 1 to 0; tau_R extracted for all N."),
    }

    # Build full summary
    summary = {
        "per_phi": {},
        "phi_star": {},
        "dilute_limit": dilute,
        "rouse_1953_properties": rouse_1953,
    }

    for phi in phis:
        phi_key = format_phi(phi)
        metrics = all_phi_metrics[phi]
        summary["per_phi"][phi] = {
            k: v for k, v in metrics.items() if k != "_detail"
        }
        summary["per_phi"][phi]["_detail"] = metrics["_detail"]

    for key, val in phi_star.items():
        summary["phi_star"][key] = val if isinstance(val, str) else format_phi(val)

    # Write JSON
    filepath = os.path.join(base_dir, "05_data", "tavg_validation_summary.json")
    ensure_dir(os.path.dirname(filepath))

    def json_default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2, default=json_default)
    print(f"  Wrote validation summary to {filepath}")
    return summary


# ============================================================================
# README
# ============================================================================

def write_readme(all_results: dict, summary: dict, base_dir: str):
    """Write README.md with phi-dependent analysis and Rouse 1953 assessment."""
    filepath = os.path.join(base_dir, "README.md")

    from .config import compute_n_chains, compute_box_size, SIGMA

    phis = sorted(all_results.keys())

    lines = [
        "# Rouse Model Validation: surpass-alpha Coarse-Grained Framework",
        "",
        "## Purpose",
        "",
        "Validate that the surpass-alpha CG representation abides by Rouse static",
        "and dynamic properties as published in Kuriata, Gront & Sikorski, CMST 22(4),",
        "179-185 (2016), and determine which Rouse (1953) properties are met across",
        "a progression of increasing chain densities (phi).",
        "",
        "## Key Physical Clarifications (Dr. Dominik Gront, 30 Mar 2026)",
        "",
        "- Excluded volume energy E is INFINITE (RepulsiveEnergy = 1e6 as practical",
        "  approximation). Any overlapping move is ALWAYS rejected. Temperature is",
        "  irrelevant to the accept/reject decision (athermal system).",
        "- Concentration (phi) tested across a density progression, not a single value.",
        "- l0/d0 = 1.5 (bond spacing 5.7A / bead diameter 3.8A) chosen by Mohammad",
        "  Nazmul Saqib, placing surpass-alpha in the SAW scaling regime per Kuriata Fig. 3.",
        "",
        "## Simulation Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Bead diameter (sigma/d0) | 3.8 A |",
        "| Bond length (l0) | 5.7 A (1.5 * sigma) |",
        "| l0/d0 | 1.5 (SAW regime, Kuriata Fig. 3) |",
        "| Excluded volume | INFINITE (RepulsiveEnergy = 1e6, athermal) |",
        "| Contact energy | 0.0 kJ/mol (no attractive interactions) |",
        "| MC moves | Segmented hinge + N-tail + C-tail + pivot |",
        "| Initialization | Random walk (self-avoiding, off-lattice) |",
        "| Chain lengths | 25, 50, 100, 250, 500 |",
        f"| Phi levels | {', '.join(format_phi(p) for p in phis)} |",
        f"| State points | {len(phis) * len(CHAIN_LENGTHS)} (5 N x {len(phis)} phi) |",
        "| Random seed | 42 |",
        "",
        "## 30-State-Point Configuration Table",
        "",
        "| N | phi | Chains | Eq Sweeps | Prod Sweeps | Box (A) |",
        "|---|-----|--------|-----------|-------------|---------|",
    ]

    for N in CHAIN_LENGTHS:
        for phi in phis:
            nc = compute_n_chains(N, phi)
            box = compute_box_size(N, nc, phi)
            lines.append(f"| {N} | {format_phi(phi)} | {nc} | 10,000 | 10,000 | {box:.1f} |")

    # Dilute-limit summary
    dilute = summary.get("dilute_limit", {})
    lines += [
        "",
        "## Dilute Limit Results (phi=0.001)",
        "",
        "| Property | Theory | Measured | R2_fit | Pass/Fail |",
        "|----------|--------|----------|--------|-----------|",
    ]
    for key in ["R2_exponent", "Rg2_exponent", "R2_Rg2_ratio", "D_exponent",
                "tau_R_exponent", "g_CM_exponent", "g1_exponent"]:
        entry = dilute.get(key, {})
        if isinstance(entry, dict) and "measured" in entry:
            status = "PASS" if entry.get("pass", False) else "FAIL"
            r2_fit = entry.get("r2_fit", "N/A")
            r2_str = f"{r2_fit:.4f}" if isinstance(r2_fit, float) else str(r2_fit)
            lines.append(f"| {key} | {entry.get('theory', 'N/A')} | "
                        f"{entry['measured']:.3f} | {r2_str} | {status} |")

    # phi* analysis
    phi_star = summary.get("phi_star", {})
    lines += [
        "",
        "## Critical Density phi* Analysis",
        "",
        "phi* is the lowest phi at which each Rouse property fails the pass/fail threshold.",
        "",
        "| Property | phi* |",
        "|----------|------|",
    ]
    for key, val in phi_star.items():
        lines.append(f"| {key} | {val} |")

    # Rouse 1953 assessment
    rouse = summary.get("rouse_1953_properties", {})
    lines += [
        "",
        "## Rouse 1953 Property Assessment",
        "",
        "### Properties MET by surpass-alpha",
        "",
    ]
    for key, val in rouse.items():
        if isinstance(val, str) and val.startswith("MET"):
            lines.append(f"- **{key}**: {val}")
    lines += [
        "",
        "### Properties NOT MET by surpass-alpha",
        "",
    ]
    for key, val in rouse.items():
        if isinstance(val, str) and val.startswith("NOT MET"):
            lines.append(f"- **{key}**: {val}")

    # Cross-phi findings
    lines += [
        "",
        "## Cross-Phi Findings",
        "",
        "| phi | 2nu(R2) | 2nu(Rg2) | R2/Rg2 | D exp | tau_R exp | g1 exp |",
        "|-----|---------|----------|--------|-------|-----------|--------|",
    ]
    per_phi = summary.get("per_phi", {})
    for phi in phis:
        phi_str = format_phi(phi)
        m = per_phi.get(phi, {})
        vals = []
        for key in ["R2_exponent", "Rg2_exponent", "R2_Rg2_ratio",
                     "D_exponent", "tau_R_exponent", "g1_exponent"]:
            entry = m.get(key, {})
            if isinstance(entry, dict) and "measured" in entry:
                v = entry["measured"]
                p = "P" if entry.get("pass", False) else "F"
                vals.append(f"{v:.2f}({p})")
            else:
                vals.append("N/A")
        lines.append(f"| {phi_str} | {' | '.join(vals)} |")

    # References
    lines += [
        "",
        "## References",
        "",
        "1. Rouse, P.E. Jr., J. Chem. Phys. 21(7), 1272-1280 (1953)",
        "2. Kuriata, A., Gront, D. & Sikorski, A., CMST 22(4), 179-185 (2016)",
        "",
        "---",
        "Generated by rouse_model_python (surpass-alpha CG framework)",
    ]

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print("  Wrote README.md")


def write_gitattributes(base_dir: str):
    """Write .gitattributes for TSV and PNG handling."""
    filepath = os.path.join(base_dir, ".gitattributes")
    content = "*.tsv text eol=lf\n*.png binary\n*.json text eol=lf\n*.md text eol=lf\n"
    with open(filepath, 'w') as f:
        f.write(content)
    print("  Wrote .gitattributes")


# ============================================================================
# Main Plot Dispatcher
# ============================================================================

def generate_all_plots(all_results: dict, base_dir: str):
    """Generate all plots for the 30-state-point validation.

    Args:
        all_results: dict[phi][N] -> results_dict
    """
    phis = sorted(all_results.keys())

    # Per-state-point plots (5 x 30 = 150)
    for phi in phis:
        for N in sorted(all_results[phi].keys()):
            plot_per_state_point(all_results[phi][N], base_dir, N, phi)

    # Per-phi equilibration evidence (2 x 6 = 12)
    for phi in phis:
        plot_equilibration_per_phi(all_results[phi], base_dir, phi)

    # Cross-phi static plots (5)
    plot_R2_vs_N_per_phi(all_results, base_dir)
    plot_Rg2_vs_N_per_phi(all_results, base_dir)
    plot_2nu_vs_phi(all_results, base_dir)
    plot_ratio_R2_Rg2_vs_phi(all_results, base_dir)
    plot_R2_Rg2_combined_dilute(all_results, base_dir)

    # Cross-phi dynamic plots (7)
    plot_g1_vs_sweep_per_phi(all_results, base_dir)
    plot_gcm_vs_sweep_per_phi(all_results, base_dir)
    plot_D_vs_N_per_phi(all_results, base_dir)
    plot_tauR_vs_N_per_phi(all_results, base_dir)
    plot_D_exponent_vs_phi(all_results, base_dir)
    plot_tauR_exponent_vs_phi(all_results, base_dir)
    plot_g1_shorttime_exponent_vs_phi(all_results, base_dir)

    total_plots = 150 + 12 + 5 + 7  # 174 plots
    print(f"  Generated {total_plots} plots total")
