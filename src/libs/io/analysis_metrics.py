"""AnalysisMetrics — per-phi scaling metrics for Rouse validation.

Holds estimators for D, tau_R, g_CM exponent, g1 short-time exponent, and
the production-averaged static observables feeding the pass/fail gates.
"""

import numpy as np
from scipy import stats as scipy_stats

from rouse_model_python.src.libs.io.plot_generator import PlotGenerator


class AnalysisMetrics:
    @staticmethod
    def estimate_diffusion_coefficient(gcm: dict) -> float:
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

    @staticmethod
    def estimate_relaxation_time(gr: dict) -> float:
        target = 1.0 / np.e
        lags = sorted(gr.keys())
        for l in lags:
            if gr[l] <= target:
                return float(l)
        if len(lags) >= 2:
            return float(lags[-1]) * 2.0
        return float(lags[-1]) if lags else 1.0

    @staticmethod
    def estimate_g1_short_time_exponent(g1: dict) -> float:
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

    @staticmethod
    def estimate_gcm_exponent(gcm: dict) -> float:
        if len(gcm) < 3:
            return 0.0
        lags = sorted(gcm.keys())
        n = len(lags)
        start = max(1, n // 2)
        x = np.array(lags[start:], dtype=float)
        y = np.array([gcm[l] for l in lags[start:]], dtype=float)
        mask = (x > 0) & (y > 0)
        if mask.sum() < 2:
            return 0.0
        slope, _, _, _, _ = scipy_stats.linregress(np.log(x[mask]), np.log(y[mask]))
        return slope

    @staticmethod
    def check_pass(measured: float, theory: float, tolerance: float) -> bool:
        return bool(abs(measured - theory) <= tolerance)

    @staticmethod
    def check_marginal(measured: float, theory: float, tolerance: float) -> int:
        diff = abs(measured - theory)
        if diff <= tolerance:
            return 2
        elif diff <= tolerance * 1.5:
            return 1
        return 0

    @staticmethod
    def production_averaged_static(results: dict):
        sweep_data = results.get('sweep_data', [])
        prod_R2 = [r2 for (sw, phase, r2, rg2, ratio) in sweep_data if phase == "production"]
        prod_Rg2 = [rg2 for (sw, phase, r2, rg2, ratio) in sweep_data if phase == "production"]
        if len(prod_R2) >= 3:
            return float(np.mean(prod_R2)), float(np.mean(prod_Rg2))
        return results['final_R2'].mean().item(), results['final_Rg2'].mean().item()

    @staticmethod
    def production_averaged_static_with_std(results: dict):
        r2_t = results['final_R2']
        rg2_t = results['final_Rg2']
        mean_R2 = r2_t.mean().item()
        mean_Rg2 = rg2_t.mean().item()
        std_R2 = r2_t.std().item() if r2_t.numel() > 1 else 0.0
        std_Rg2 = rg2_t.std().item() if rg2_t.numel() > 1 else 0.0
        sweep_data = results.get('sweep_data', [])
        prod_R2 = [r2 for (sw, phase, r2, rg2, ratio) in sweep_data if phase == "production"]
        prod_Rg2 = [rg2 for (sw, phase, r2, rg2, ratio) in sweep_data if phase == "production"]
        if len(prod_R2) >= 3:
            mean_R2 = float(np.mean(prod_R2))
            mean_Rg2 = float(np.mean(prod_Rg2))
        return mean_R2, mean_Rg2, std_R2, std_Rg2

    @classmethod
    def compute_phi_metrics(cls, phi_results: dict) -> dict:
        Ns = sorted(phi_results.keys())
        mean_R2 = {}
        mean_Rg2 = {}
        for n in Ns:
            r2, rg2 = cls.production_averaged_static(phi_results[n])
            mean_R2[n] = r2
            mean_Rg2[n] = rg2
        ratios = {n: mean_R2[n] / mean_Rg2[n] if mean_Rg2[n] > 0 else 0.0 for n in Ns}

        slope_R2, _, r2_R2 = PlotGenerator.power_law_fit(Ns, [mean_R2[n] for n in Ns])
        slope_Rg2, _, r2_Rg2 = PlotGenerator.power_law_fit(Ns, [mean_Rg2[n] for n in Ns])

        Ds = {n: cls.estimate_diffusion_coefficient(phi_results[n]['gcm']) for n in Ns}
        valid_D = [(n, Ds[n]) for n in Ns if Ds[n] > 0]
        diff_slope, diff_r2 = 0.0, 0.0
        if valid_D:
            diff_slope, _, diff_r2 = PlotGenerator.power_law_fit(
                [v[0] for v in valid_D], [v[1] for v in valid_D])

        taus = {n: cls.estimate_relaxation_time(phi_results[n]['gr']) for n in Ns}
        valid_tau = [(n, taus[n]) for n in Ns if taus[n] > 0]
        relax_slope, relax_r2 = 0.0, 0.0
        if valid_tau:
            relax_slope, _, relax_r2 = PlotGenerator.power_law_fit(
                [v[0] for v in valid_tau], [v[1] for v in valid_tau])

        gcm_exponents = {n: cls.estimate_gcm_exponent(phi_results[n]['gcm']) for n in Ns}
        mean_gcm_exp = (np.mean([v for v in gcm_exponents.values() if v > 0])
                        if any(v > 0 for v in gcm_exponents.values()) else 0.0)

        g1_exponents = {n: cls.estimate_g1_short_time_exponent(phi_results[n]['g1']) for n in Ns}
        mean_g1_exp = (np.mean([v for v in g1_exponents.values() if v > 0])
                       if any(v > 0 for v in g1_exponents.values()) else 0.0)

        mean_ratio = float(np.mean(list(ratios.values())))

        return {
            "R2_exponent": {"measured": round(float(slope_R2), 3), "theory": 1.18,
                            "r2_fit": round(float(r2_R2), 4),
                            "pass": cls.check_pass(slope_R2, 1.18, 0.10)},
            "Rg2_exponent": {"measured": round(float(slope_Rg2), 3), "theory": 1.18,
                             "r2_fit": round(float(r2_Rg2), 4),
                             "pass": cls.check_pass(slope_Rg2, 1.18, 0.10)},
            "R2_Rg2_ratio": {"measured": round(float(mean_ratio), 3), "theory": 6.25,
                             "pass": cls.check_pass(mean_ratio, 6.25, 1.25)},
            "D_exponent": {"measured": round(float(diff_slope), 3), "theory": -1.00,
                           "r2_fit": round(float(diff_r2), 4),
                           "pass": cls.check_pass(diff_slope, -1.00, 0.10)},
            "tau_R_exponent": {"measured": round(float(relax_slope), 3), "theory": 2.18,
                               "r2_fit": round(float(relax_r2), 4),
                               "pass": cls.check_pass(relax_slope, 2.18, 0.20)},
            "g_CM_exponent": {"measured": round(float(mean_gcm_exp), 3), "theory": 1.00,
                              "pass": cls.check_pass(mean_gcm_exp, 1.00, 0.10)},
            "g1_exponent": {"measured": round(float(mean_g1_exp), 3), "theory": 0.50,
                            "pass": cls.check_pass(mean_g1_exp, 0.50, 0.15)},
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
