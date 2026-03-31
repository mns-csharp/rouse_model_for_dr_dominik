"""
Fast simulation using numpy + numba for the MC sweep hot path.

Drop-in replacement for simulation.py's RouseSimulation.
Uses fast_sweep.fast_perform_sweep() instead of multistep_mc.perform_sweep().
Positions stored as numpy arrays; PyTorch used only for initialization and
observable computation (which is not on the hot path).
"""

import time
import numpy as np
import torch
from tqdm import tqdm

from .config import SimulationConfig
from .chain import ChainState
from .fast_energy import FastEnergyComputer
from .fast_sweep import fast_perform_sweep, fast_perform_pivot_phase
from .observables import StaticObservables, DynamicAccumulator, SweepTracker
from .number_space import NumberSpace
from .simulation import SimulationStats


class FastRouseSimulation:
    """
    Rouse-model MC simulation using numba-optimized sweep.

    Same physics and algorithm as RouseSimulation, but with numpy/numba
    for the MC sweep hot path. Observables still use PyTorch via NumberSpace.
    """

    def __init__(self, cfg: SimulationConfig, snapshot_collector=None):
        self.cfg = cfg
        self.ns = NumberSpace.from_config(cfg)
        self.state = ChainState(cfg)
        self.fast_energy = FastEnergyComputer(cfg)
        self.gen = cfg.get_torch_gen()
        self.stats = SimulationStats()
        self.snapshot_collector = snapshot_collector
        self.rng = np.random.RandomState(cfg.seed)

        # Observables
        self.dynamic_accum = DynamicAccumulator(cfg, self.ns)
        self.sweep_tracker = SweepTracker(cfg, self.ns)

        self.final_R2 = None
        self.final_Rg2 = None

        # Numpy position array (the main working copy)
        self._pos_np = None

    def initialize(self):
        """Initialize chain positions using PyTorch random walk, then copy to numpy."""
        self.state.initialize_random_walk(self.gen)
        self.state.wrap_all()
        # Copy to numpy for fast sweep
        self._pos_np = self.state.positions.cpu().numpy().copy()
        print(f"  Initialized {self.cfg.n_chains} chains of N={self.cfg.N} "
              f"(box={self.cfg.box_size:.1f} A, fast mode)", flush=True)

    def _sync_torch_from_numpy(self):
        """Copy numpy positions back to PyTorch state for observable computation."""
        self.state.positions.copy_(torch.from_numpy(self._pos_np))

    def run_equilibration(self):
        """Run equilibration sweeps with fast path."""
        cfg = self.cfg
        total = cfg.eq_sweeps
        seg_info = self.state.segments

        pbar = tqdm(range(total), desc="  Eq", unit="sw",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        for sweep in pbar:
            fast_perform_sweep(self._pos_np, seg_info, self.fast_energy,
                               cfg, self.stats, self.rng)

            if sweep % max(1, total // 50) == 0:
                # Sync to torch for observable computation
                self._sync_torch_from_numpy()
                self.sweep_tracker.record(self.state, sweep, "equilibration")

            if self.snapshot_collector is not None:
                self._sync_torch_from_numpy()
                self.snapshot_collector.capture(self.state.positions, sweep, "eq")

            pbar.set_postfix_str(self.stats.report(), refresh=False)

        pbar.close()
        dt = pbar.format_dict.get('elapsed', 0)
        if dt > 0:
            print(f"  Equilibration done in {dt:.1f}s ({total/dt:.2f} sweeps/s)")

    def run_production(self):
        """Run production sweeps with observable collection.

        Production uses LOCAL moves only (no pivots) to give correct Rouse
        dynamics (D ~ N^-1, tau_R ~ N^(1+2nu)).  Pivots are NOT used during
        production because they:
          1. Add CM displacement ~ N^(2nu), breaking D ~ N^-1
          2. Randomise the end-to-end vector, making tau_R artificially short
        Equilibration (which uses all moves including pivots) ensures the
        starting configuration is properly sampled.

        Per-bead cumulative displacement is tracked in numpy space (zero-copy)
        to avoid per-sweep torch sync overhead.
        """
        cfg = self.cfg
        ns = self.ns
        total = cfg.prod_sweeps
        sample_interval = cfg.sample_interval
        seg_info = self.state.segments

        # Cumulative per-bead displacement tracked in numpy (fast, no sync)
        cum_seg_disp_np = np.zeros_like(self._pos_np)  # [n_chains, N, 3]
        box = cfg.box_size
        inv_box = 1.0 / box

        pbar = tqdm(range(total), desc="  Prod", unit="sw",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        for sweep in pbar:
            # Save positions before segment moves
            pos_before = self._pos_np.copy()

            # LOCAL moves only — this IS the Rouse time step
            fast_perform_sweep(self._pos_np, seg_info, self.fast_energy,
                               cfg, self.stats, self.rng, skip_pivot=True)

            # Per-bead displacement with MIC wrapping (numpy, no torch sync)
            delta = self._pos_np - pos_before
            delta -= box * np.round(delta * inv_box)
            cum_seg_disp_np += delta

            if sweep % max(1, total // 50) == 0:
                self._sync_torch_from_numpy()
                self.sweep_tracker.record(
                    self.state, cfg.eq_sweeps + sweep, "production")

            if sweep % sample_interval == 0:
                self._sync_torch_from_numpy()
                cum_seg_disp_torch = torch.from_numpy(
                    cum_seg_disp_np.copy()).to(cfg.dtype)
                self.dynamic_accum.record_snapshot(
                    self.state, sweep, cum_bead_disp=cum_seg_disp_torch)

            if self.snapshot_collector is not None:
                self._sync_torch_from_numpy()
                self.snapshot_collector.capture(
                    self.state.positions, cfg.eq_sweeps + sweep, "prod")

            pbar.set_postfix_str(self.stats.report(), refresh=False)

        pbar.close()
        dt = pbar.format_dict.get('elapsed', 0)

        # Final snapshot
        self._sync_torch_from_numpy()
        positions = self.state.positions
        self.final_R2 = StaticObservables.compute_R2(positions, ns)
        self.final_Rg2 = StaticObservables.compute_Rg2(positions, ns)

        mean_R2 = self.final_R2.mean().item()
        mean_Rg2 = self.final_Rg2.mean().item()
        ratio = mean_R2 / mean_Rg2 if mean_Rg2 > 0 else 0
        if dt > 0:
            print(f"  Production done in {dt:.1f}s ({total/dt:.2f} sweeps/s)")
        print(f"  <R²>={mean_R2:.1f}, <Rg²>={mean_Rg2:.1f}, R²/Rg²={ratio:.2f}")

    def run(self):
        """Run full simulation: init + equilibration + production."""
        print(f"\n{'='*60}")
        print(f"Running N={self.cfg.N}, {self.cfg.n_chains} chains, "
              f"fast mode (numba+numpy)")
        print(f"{'='*60}")

        self.initialize()
        self.run_equilibration()
        self.run_production()

        print(f"Final acceptance rates: {self.stats.report()}")
        return self.get_results()

    def get_results(self) -> dict:
        """Collect all simulation results."""
        return {
            'N': self.cfg.N,
            'n_chains': self.cfg.n_chains,
            'final_R2': self.final_R2,
            'final_Rg2': self.final_Rg2,
            'g1': self.dynamic_accum.get_g1(),
            'gcm': self.dynamic_accum.get_gcm(),
            'gr': self.dynamic_accum.get_gr(),
            'sweep_data': self.sweep_tracker.records,
            'stats': self.stats,
        }
