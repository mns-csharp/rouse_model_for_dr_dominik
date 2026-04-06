"""
GPU fast simulation: full MC simulation with GPU-resident positions.

Drop-in replacement for FastRouseSimulation. Same physics and interface,
but the MC sweep runs on GPU via CuPy RawKernels. Positions stay on GPU
as PyTorch CUDA tensors; observables use existing PyTorch code directly.
"""

import numpy as np
import torch
from tqdm import tqdm

from .config import SimulationConfig
from .chain import ChainState
from .gpu_sweep import GPUFastSweep, gpu_perform_sweep
from .observables import StaticObservables, DynamicAccumulator, SweepTracker
from .number_space import NumberSpace
from .simulation import SimulationStats


class GPURouseSimulation:
    """
    Rouse-model MC simulation using GPU-accelerated sweep.

    Same physics and algorithm as FastRouseSimulation, but with positions
    on GPU and the sweep hot path running via CuPy CUDA kernels.
    Observables use existing PyTorch code directly (positions are CUDA tensors).
    """

    def __init__(self, cfg: SimulationConfig, snapshot_collector=None):
        self.cfg = cfg
        self.ns = NumberSpace.from_config(cfg)
        self.state = ChainState(cfg)
        self.gen = cfg.get_torch_gen()
        self.stats = SimulationStats()
        self.snapshot_collector = snapshot_collector
        self.rng = np.random.RandomState(cfg.seed)

        # GPU sweep engine (compiled kernels, cell list, buffers)
        self.device = cfg.get_torch_device()
        self.sweep_engine = GPUFastSweep(cfg, self.device)

        # Observables
        self.dynamic_accum = DynamicAccumulator(cfg, self.ns)
        self.sweep_tracker = SweepTracker(cfg, self.ns)

        self.final_R2 = None
        self.final_Rg2 = None

    def initialize(self):
        """Initialize chain positions using PyTorch random walk."""
        self.state.initialize_random_walk(self.gen)
        self.state.wrap_all()
        # Ensure positions are on the target GPU device
        self.state.positions = self.state.positions.to(self.device)
        print(f"  Initialized {self.cfg.n_chains} chains of N={self.cfg.N} "
              f"(box={self.cfg.box_size:.1f} A, GPU fast mode, {self.device})",
              flush=True)

    def run_equilibration(self):
        """Run equilibration sweeps with GPU fast path."""
        cfg = self.cfg
        total = cfg.eq_sweeps
        seg_info = self.state.segments

        pbar = tqdm(range(total), desc="  Eq", unit="sw",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        for sweep in pbar:
            gpu_perform_sweep(self.state.positions, seg_info,
                              self.sweep_engine, cfg, self.stats, self.rng)

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
        """Run production sweeps with observable collection.

        Production uses LOCAL moves only (no pivots) to give correct Rouse
        dynamics (D ~ N^-1, tau_R ~ N^(1+2nu)).  Pivots are NOT used during
        production because they:
          1. Add CM displacement ~ N^(2nu), breaking D ~ N^-1
          2. Randomise the end-to-end vector, making tau_R artificially short
        Equilibration (which uses all moves including pivots) ensures the
        starting configuration is properly sampled.

        Per-bead cumulative displacement is tracked on GPU.
        """
        cfg = self.cfg
        ns = self.ns
        total = cfg.prod_sweeps
        sample_interval = cfg.sample_interval
        seg_info = self.state.segments

        # Cumulative per-bead displacement tracked on GPU
        cum_disp = torch.zeros_like(self.state.positions)  # [n_chains, N, 3] on GPU
        box = cfg.box_size
        inv_box = 1.0 / box

        pbar = tqdm(range(total), desc="  Prod", unit="sw",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        for sweep in pbar:
            # Save positions before sweep
            pos_before = self.state.positions.clone()

            # LOCAL moves only — this IS the Rouse time step
            gpu_perform_sweep(self.state.positions, seg_info,
                              self.sweep_engine, cfg, self.stats, self.rng,
                              skip_pivot=True)

            # Per-bead displacement with MIC wrapping (on GPU)
            delta = self.state.positions - pos_before
            delta -= box * torch.round(delta * inv_box)
            cum_disp += delta

            if sweep % max(1, total // 50) == 0:
                self.sweep_tracker.record(
                    self.state, cfg.eq_sweeps + sweep, "production")

            if sweep % sample_interval == 0:
                cum_disp_copy = cum_disp.clone().to(cfg.dtype)
                self.dynamic_accum.record_snapshot(
                    self.state, sweep, cum_bead_disp=cum_disp_copy)

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
              f"GPU fast mode (CuPy+CUDA)")
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
