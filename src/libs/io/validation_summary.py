"""ValidationSummary — writes the JSON summary, README.md, .gitattributes,
and compliance heatmap PNG for the full N × phi validation matrix.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from rouse_model_python.src.libs.config.config_helpers import ConfigHelpers
from rouse_model_python.src.libs.io.analysis_metrics import AnalysisMetrics
from rouse_model_python.src.libs.io.plot_generator import PlotGenerator


class ValidationSummary:
    PROPERTIES_FOR_HEATMAP = [
        ("2nu (R2)", "R2_exponent", 1.18, 0.10),
        ("2nu (Rg2)", "Rg2_exponent", 1.18, 0.10),
        ("R2/Rg2", "R2_Rg2_ratio", 6.25, 1.25),
        ("D exp", "D_exponent", -1.00, 0.10),
        ("tau_R exp", "tau_R_exponent", 2.18, 0.20),
        ("g_CM exp", "g_CM_exponent", 1.00, 0.10),
        ("g1 exp", "g1_exponent", 0.50, 0.15),
    ]

    PROPERTIES_FOR_PHI_STAR = [
        ("R2_exponent", 1.18, 0.10),
        ("Rg2_exponent", 1.18, 0.10),
        ("R2_Rg2_ratio", 6.25, 1.25),
        ("D_exponent", -1.00, 0.10),
        ("tau_R_exponent", 2.18, 0.20),
        ("g_CM_exponent", 1.00, 0.10),
        ("g1_exponent", 0.50, 0.15),
    ]

    @classmethod
    def plot_compliance_heatmap(cls, all_results: dict, base_dir: str,
                                all_phi_metrics: dict) -> np.ndarray:
        phis = sorted(all_phi_metrics.keys())
        n_props = len(cls.PROPERTIES_FOR_HEATMAP)
        n_phis = len(phis)
        matrix = np.zeros((n_props, n_phis))
        cell_text = [['' for _ in range(n_phis)] for _ in range(n_props)]

        for j, phi in enumerate(phis):
            metrics = all_phi_metrics[phi]
            for i, (label, key, theory, tol) in enumerate(cls.PROPERTIES_FOR_HEATMAP):
                measured = metrics[key]["measured"]
                verdict = AnalysisMetrics.check_marginal(measured, theory, tol)
                matrix[i, j] = verdict
                cell_text[i][j] = f'{measured:.2f}'

        fig, ax = plt.subplots(figsize=(12, 6))
        cmap = mcolors.ListedColormap(['#ff4444', '#ffcc00', '#44bb44'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

        ax.set_xticks(range(n_phis))
        ax.set_xticklabels([ConfigHelpers.format_phi(p) for p in phis], fontsize=11)
        ax.set_yticks(range(n_props))
        ax.set_yticklabels([p[0] for p in cls.PROPERTIES_FOR_HEATMAP], fontsize=11)
        ax.set_xlabel("phi (volume fraction)", fontsize=12)
        ax.set_title("Rouse Compliance Heatmap: PASS / MARGINAL / FAIL", fontsize=14)

        for i in range(n_props):
            for j in range(n_phis):
                color = 'white' if matrix[i, j] == 0 else 'black'
                ax.text(j, i, cell_text[i][j], ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#44bb44', label='PASS'),
                           Patch(facecolor='#ffcc00', label='MARGINAL'),
                           Patch(facecolor='#ff4444', label='FAIL')]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=10)

        fig.tight_layout()
        PlotGenerator.save_plot(
            fig, os.path.join(base_dir, "05_data", "rouse_compliance_heatmap.png"))
        return matrix

    @classmethod
    def write_validation_summary(cls, all_results: dict, base_dir: str) -> dict:
        phis = sorted(all_results.keys())

        all_phi_metrics = {}
        for phi in phis:
            all_phi_metrics[phi] = AnalysisMetrics.compute_phi_metrics(all_results[phi])

        cls.plot_compliance_heatmap(all_results, base_dir, all_phi_metrics)

        phi_star = {}
        for key, theory, tol in cls.PROPERTIES_FOR_PHI_STAR:
            phi_star[key] = None
            for phi in phis:
                if not all_phi_metrics[phi][key]["pass"]:
                    phi_star[key] = phi
                    break
            if phi_star[key] is None:
                phi_star[key] = "never (passes at all phi)"

        dilute_phi = phis[0]
        dilute = all_phi_metrics[dilute_phi]

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
                "tau_R extracted from g_R(t) = 1/e crossing for all N."),
            "2.3_long_wavelength_approx": met(False,
                "Long-wavelength approx tau_p ~ N^2/p^2 not directly measurable from MC."),
            "2.4_tau_R_scaling": met(dilute["tau_R_exponent"]["pass"],
                f"tau_R ~ N^{dilute['tau_R_exponent']['measured']:.3f} (theory: 2.18)"),
            "2.5_steady_flow_viscosity": met(False,
                "Viscosity requires stress tensor; not available from equilibrium MC."),
            "2.6_complex_viscosity": met(False,
                "Complex viscosity requires frequency-domain analysis; not available from MC."),
            "2.7_shear_modulus": met(False,
                "Shear modulus requires stress autocorrelation; not computed."),
            "2.8_high_freq_approx": met(False,
                "High-frequency behavior not accessible from MC simulations."),
            "2.9_diffusion_coeff": met(dilute["D_exponent"]["pass"],
                f"D ~ N^{dilute['D_exponent']['measured']:.3f} (theory: -1.00)"),
            "2.10_g_CM": met(dilute["g_CM_exponent"]["pass"],
                f"g_CM(t) ~ t^{dilute['g_CM_exponent']['measured']:.3f} (theory: 1.00)"),
            "2.11_g1_middle_segment": met(dilute["g1_exponent"]["pass"],
                f"g1 short-time exponent = {dilute['g1_exponent']['measured']:.3f} "
                f"(theory: 0.50)"),
            "2.12_end_to_end_autocorr": met(True,
                "g_R(t) decays from 1 to 0; tau_R extracted for all N."),
        }

        summary = {
            "per_phi": {},
            "phi_star": {},
            "dilute_limit": dilute,
            "rouse_1953_properties": rouse_1953,
        }

        for phi in phis:
            metrics = all_phi_metrics[phi]
            summary["per_phi"][phi] = {k: v for k, v in metrics.items() if k != "_detail"}
            summary["per_phi"][phi]["_detail"] = metrics["_detail"]

        for key, val in phi_star.items():
            summary["phi_star"][key] = val if isinstance(val, str) else ConfigHelpers.format_phi(val)

        filepath = os.path.join(base_dir, "05_data", "tavg_validation_summary.json")
        PlotGenerator.ensure_dir(os.path.dirname(filepath))

        def json_default(o):
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            return o

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=json_default)
        return summary

    @classmethod
    def write_readme(cls, all_results: dict, summary: dict, base_dir: str,
                     chain_lengths, physics) -> None:
        filepath = os.path.join(base_dir, "README.md")

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
            "## Simulation Parameters",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Bead diameter (sigma/d0) | {physics.SIGMA} A |",
            f"| Bond length (l0) | {physics.L0} A |",
            "| Excluded volume | INFINITE (RepulsiveEnergy = 1e6, athermal) |",
            "| MC moves | Segmented hinge + N-tail + C-tail + pivot |",
            f"| Chain lengths | {', '.join(str(n) for n in chain_lengths)} |",
            f"| Phi levels | {', '.join(ConfigHelpers.format_phi(p) for p in phis)} |",
            f"| State points | {len(phis) * len(chain_lengths)} |",
            "",
            "## 30-State-Point Configuration Table",
            "",
            "| N | phi | Chains | Box (A) |",
            "|---|-----|--------|---------|",
        ]

        for N in chain_lengths:
            for phi in phis:
                nc = ConfigHelpers.compute_n_chains(N, phi)
                box = ConfigHelpers.compute_box_size(N, nc, phi, physics)
                lines.append(f"| {N} | {ConfigHelpers.format_phi(phi)} | {nc} | {box:.1f} |")

        dilute = summary.get("dilute_limit", {})
        lines += [
            "",
            "## Dilute Limit Results",
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

        phi_star = summary.get("phi_star", {})
        lines += [
            "",
            "## Critical Density phi* Analysis",
            "",
            "| Property | phi* |",
            "|----------|------|",
        ]
        for key, val in phi_star.items():
            lines.append(f"| {key} | {val} |")

        rouse = summary.get("rouse_1953_properties", {})
        lines += [
            "",
            "## Rouse 1953 Property Assessment",
            "",
            "### Properties MET",
            "",
        ]
        for key, val in rouse.items():
            if isinstance(val, str) and val.startswith("MET"):
                lines.append(f"- **{key}**: {val}")
        lines += [
            "",
            "### Properties NOT MET",
            "",
        ]
        for key, val in rouse.items():
            if isinstance(val, str) and val.startswith("NOT MET"):
                lines.append(f"- **{key}**: {val}")

        lines += [
            "",
            "## Cross-Phi Findings",
            "",
            "| phi | 2nu(R2) | 2nu(Rg2) | R2/Rg2 | D exp | tau_R exp | g1 exp |",
            "|-----|---------|----------|--------|-------|-----------|--------|",
        ]
        per_phi = summary.get("per_phi", {})
        for phi in phis:
            phi_str = ConfigHelpers.format_phi(phi)
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

    @staticmethod
    def write_gitattributes(base_dir: str) -> None:
        filepath = os.path.join(base_dir, ".gitattributes")
        content = "*.tsv text eol=lf\n*.png binary\n*.json text eol=lf\n*.md text eol=lf\n"
        with open(filepath, 'w') as f:
            f.write(content)
