"""GPURouseSimulation — GPU-resident Rouse MC simulation orchestrator.

Drop-in replacement for FastRouseSimulation with positions on CUDA and the
MC sweep running via CuPy RawKernels. Observables stay in PyTorch (no sync
needed — positions are already CUDA tensors).
"""

import torch
from tqdm import tqdm
import numpy as np

from rouse_model_python.src.libs.chain.chain_state import ChainState
from rouse_model_python.src.libs.chain.bond_rescaler import rescale_bonds_torch
from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.number_space.number_space import NumberSpace
from rouse_model_python.src.libs.observables.dynamic_accumulator import DynamicAccumulator
from rouse_model_python.src.libs.observables.static_observables import StaticObservables
from rouse_model_python.src.libs.observables.sweep_tracker import SweepTracker
from rouse_model_python.src.libs.paths.gpu.gpu_fast_sweep import GPUFastSweep
from rouse_model_python.src.libs.simulation.simulation_stats import SimulationStats


class GPURouseSimulation:
    def __init__(self, cfg: SimulationConfig, snapshot_collector=None):
        self.cfg = cfg
        self.ns = NumberSpace.from_config(cfg)
        self.state = ChainState(cfg)
        self.gen = cfg.get_torch_gen()
        self.stats = SimulationStats()
        self.snapshot_collector = snapshot_collector
        self.rng = np.random.RandomState(cfg.seed)

        self.device = cfg.get_torch_device()
        self.sweep_engine = GPUFastSweep(cfg, self.device)

        self.dynamic_accum = DynamicAccumulator(cfg, self.ns)
        self.sweep_tracker = SweepTracker(cfg, self.ns)

        self.final_R2 = None
        self.final_Rg2 = None

    def initialize(self) -> None:
        self.state.initialize_random_walk(self.gen)
        self.state.wrap_all()
        self.state.positions = self.state.positions.to(self.device)
        print(f"  Initialized {self.cfg.n_chains} chains of N={self.cfg.N} "
              f"(box={self.cfg.box_size:.1f} A, GPU fast mode, {self.device})",
              flush=True)

    def run_equilibration(self) -> None:
        cfg = self.cfg
        total = cfg.eq_sweeps
        seg_info = self.state.segments

        pbar = tqdm(range(total), desc="  Eq", unit="sw",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        for sweep in pbar:
            self.sweep_engine.perform_sweep(self.state.positions, seg_info,
                                            self.stats, self.rng)

            if cfg.n_small_steps > 0 and (sweep + 1) % cfg.n_small_steps == 0:
                drift = rescale_bonds_torch(
                    self.state.positions, cfg.l0, cfg.box_size, cfg.half_box)
                self.stats.rescale_count += 1
                self.stats.rescale_last_drift = drift
                if drift > self.stats.rescale_max_drift:
                    self.stats.rescale_max_drift = drift

            if sweep % max(1, total // 50) == 0:
                self.sweep_tracker.record(self.state, sweep, "equilibration")

            sc = self.snapshot_collector
            if sc is not None and getattr(sc, "should_capture", lambda *_: True)(sweep, "eq"):
                sc.capture(self.state.positions, sweep, "eq")

            pbar.set_postfix_str(self.stats.report(), refresh=False)

        pbar.close()
        dt = pbar.format_dict.get('elapsed', 0)
        if dt > 0:
            print(f"  Equilibration done in {dt:.1f}s ({total/dt:.2f} sweeps/s)")

    def run_production(self) -> None:
        cfg = self.cfg
        ns = self.ns
        total = cfg.prod_sweeps
        sample_interval = cfg.sample_interval
        seg_info = self.state.segments

        cum_disp = torch.zeros_like(self.state.positions)
        box = cfg.box_size
        inv_box = 1.0 / box

        pbar = tqdm(range(total), desc="  Prod", unit="sw",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        for sweep in pbar:
            pos_before = self.state.positions.clone()

            self.sweep_engine.perform_sweep(self.state.positions, seg_info,
                                            self.stats, self.rng, skip_pivot=True)

            if cfg.n_small_steps > 0 and (sweep + 1) % cfg.n_small_steps == 0:
                drift = rescale_bonds_torch(
                    self.state.positions, cfg.l0, cfg.box_size, cfg.half_box)
                self.stats.rescale_count += 1
                self.stats.rescale_last_drift = drift
                if drift > self.stats.rescale_max_drift:
                    self.stats.rescale_max_drift = drift

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

            sc = self.snapshot_collector
            if sc is not None and getattr(sc, "should_capture", lambda *_: True)(cfg.eq_sweeps + sweep, "prod"):
                sc.capture(self.state.positions, cfg.eq_sweeps + sweep, "prod")

            pbar.set_postfix_str(self.stats.report(), refresh=False)

        pbar.close()
        dt = pbar.format_dict.get('elapsed', 0)

        positions = self.state.positions
        self.final_R2 = StaticObservables.compute_R2(positions, ns)
        self.final_Rg2 = StaticObservables.compute_Rg2(positions, ns)

        mean_R2 = self.final_R2.mean().item()
        mean_Rg2 = self.final_Rg2.mean().item()
        ratio = mean_R2 / mean_Rg2 if mean_Rg2 > 0 else 0
        if dt > 0:
            print(f"  Production done in {dt:.1f}s ({total/dt:.2f} sweeps/s)")
        print(f"  <R²>={mean_R2:.1f}, <Rg²>={mean_Rg2:.1f}, R²/Rg²={ratio:.2f}")

    def run(self) -> dict:
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
