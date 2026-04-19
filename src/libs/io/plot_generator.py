"""PlotGenerator — plot primitives and helpers (figure setup, save, power-law fit)."""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats


class PlotGenerator:
    PHI_COLORS = {
        0.001: '#1f77b4',
        0.01:  '#ff7f0e',
        0.05:  '#2ca02c',
        0.10:  '#d62728',
        0.20:  '#9467bd',
        0.30:  '#8c564b',
    }

    @staticmethod
    def ensure_dir(path: str) -> None:
        if path:
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def setup_plot(xlabel: str, ylabel: str, title: str, loglog: bool = True):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        if loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        return fig, ax

    @staticmethod
    def save_plot(fig, filepath: str) -> None:
        PlotGenerator.ensure_dir(os.path.dirname(filepath))
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    @classmethod
    def phi_color(cls, phi: float) -> str:
        return cls.PHI_COLORS.get(phi, 'black')

    @staticmethod
    def power_law_fit(x, y):
        mask = (np.array(x) > 0) & (np.array(y) > 0)
        lx = np.log(np.array(x)[mask])
        ly = np.log(np.array(y)[mask])
        if len(lx) < 2:
            return 0.0, 0.0, 0.0
        slope, intercept, r, p, se = scipy_stats.linregress(lx, ly)
        return slope, intercept, r ** 2
