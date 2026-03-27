"""
Main simulation orchestrator: initialization, equilibration, production.

The MC sweep algorithm lives in multistep_mc.py. This module handles
the simulation lifecycle and observable collection.
"""

import time
import torch
from tqdm import tqdm
from .config import SimulationConfig
from .chain import ChainState
from .energy import EnergyComputer
from .multistep_mc import perform_sweep, metropolis_accept
from .observables import (StaticObservables, DynamicAccumulator, SweepTracker)
from .number_space import NumberSpace


class SimulationStats:
    """Tracks acceptance rates for each move type."""

    def __init__(self):
        self.hinge_attempted = 0
        self.hinge_accepted = 0
        self.tail_attempted = 0
        self.tail_accepted = 0
        self.pivot_attempted = 0
        self.pivot_accepted = 0

    def record(self, move_type: str, accepted: bool):
        if move_type == 'hinge':
            self.hinge_attempted += 1
            if accepted:
                self.hinge_accepted += 1
        elif move_type == 'tail':
            self.tail_attempted += 1
            if accepted:
                self.tail_accepted += 1
        elif move_type == 'pivot':
            self.pivot_attempted += 1
            if accepted:
                self.pivot_accepted += 1

    def report(self) -> str:
        def rate(a, t):
            return f"{100.0 * a / t:.1f}%" if t > 0 else "N/A"
        return (f"Hinge: {rate(self.hinge_accepted, self.hinge_attempted)} "
                f"({self.hinge_accepted}/{self.hinge_attempted}), "
                f"Tail: {rate(self.tail_accepted, self.tail_attempted)} "
                f"({self.tail_accepted}/{self.tail_attempted}), "
                f"Pivot: {rate(self.pivot_accepted, self.pivot_attempted)} "
                f"({self.pivot_accepted}/{self.pivot_attempted})")


class RouseSimulation:
    """
    Full Rouse-model MC simulation for a given chain length.

    Runs equilibration + production, collecting observables during production.
    NumberSpace is created once and shared across all components.
    """

    def __init__(self, cfg: SimulationConfig, snapshot_collector=None):
        self.cfg = cfg
        self.ns = NumberSpace.from_config(cfg)
        self.state = ChainState(cfg)
        self.energy_comp = EnergyComputer(cfg, self.ns)
        self.gen = cfg.get_torch_gen()
        self.stats = SimulationStats()
        self.snapshot_collector = snapshot_collector

        # Observables (pass NumberSpace for MIC operations)
        self.dynamic_accum = DynamicAccumulator(cfg, self.ns)
        self.sweep_tracker = SweepTracker(cfg, self.ns)

        # Final snapshot observables
        self.final_R2 = None
        self.final_Rg2 = None

    def initialize(self):
        """Initialize chain positions (random walk for faster equilibration)."""
        self.state.initialize_random_walk(self.gen)
        self.state.wrap_all()
        print(f"  Initialized {self.cfg.n_chains} chains of N={self.cfg.N} "
              f"(box={self.cfg.box_size:.1f} A)", flush=True)

    def run_equilibration(self):
        """Run equilibration sweeps (records R²/Rg² for equilibration evidence, but no dynamic observables)."""
        cfg = self.cfg
        total = cfg.eq_sweeps
        pbar = tqdm(range(total), desc="  Eq", unit="sw",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        for sweep in pbar:
            perform_sweep(self.state, self.energy_comp, self.gen, cfg, self.stats)

            if sweep % max(1, total // 50) == 0:
                self.sweep_tracker.record(self.state, sweep, "equilibration")

            if self.snapshot_collector is not None:
                self.snapshot_collector.capture(self.state.positions, sweep, "eq")

            pbar.set_postfix_str(self.stats.report(), refresh=False)

        pbar.close()
        dt = pbar.format_dict.get('elapsed', 0)
        if dt > 0:
            print(f"  Equilibration done in {dt:.1f}s ({total/dt:.2f} sweeps/s)")

    def run_production(self):
        """Run production sweeps with observable collection."""
        cfg = self.cfg
        ns = self.ns
        total = cfg.prod_sweeps
        sample_interval = cfg.sample_interval
        pbar = tqdm(range(total), desc="  Prod", unit="sw",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        for sweep in pbar:
            perform_sweep(self.state, self.energy_comp, self.gen, cfg, self.stats)

            if sweep % max(1, total // 50) == 0:
                self.sweep_tracker.record(
                    self.state, cfg.eq_sweeps + sweep, "production")

            if sweep % sample_interval == 0:
                self.dynamic_accum.record_snapshot(self.state, sweep)

            if self.snapshot_collector is not None:
                self.snapshot_collector.capture(
                    self.state.positions, cfg.eq_sweeps + sweep, "prod")

            pbar.set_postfix_str(self.stats.report(), refresh=False)

        pbar.close()
        dt = pbar.format_dict.get('elapsed', 0)

        # Final snapshot
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
              f"device={self.cfg.device}")
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
