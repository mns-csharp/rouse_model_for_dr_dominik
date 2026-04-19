"""FastRouseSimulation — numpy/numba Rouse MC simulation orchestrator.

Drop-in replacement for RouseSimulation (torch path). Uses FastSweep for the
hot-path MC sweep and PyTorch only for initialization and observable
computation (off the hot path).
"""

import numpy as np
import torch
from tqdm import tqdm

from rouse_model_python.src.libs.chain.chain_state import ChainState
from rouse_model_python.src.libs.chain.bond_rescaler import rescale_bonds_numpy
from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.number_space.number_space import NumberSpace
from rouse_model_python.src.libs.observables.dynamic_accumulator import DynamicAccumulator
from rouse_model_python.src.libs.observables.static_observables import StaticObservables
from rouse_model_python.src.libs.observables.sweep_tracker import SweepTracker
from rouse_model_python.src.libs.paths.fast.fast_energy_computer import FastEnergyComputer
from rouse_model_python.src.libs.paths.fast.fast_sweep import FastSweep
from rouse_model_python.src.libs.simulation.simulation_stats import SimulationStats


class FastRouseSimulation:
    def __init__(self, cfg: SimulationConfig, snapshot_collector=None):
        self.cfg = cfg
        self.ns = NumberSpace.from_config(cfg)
        self.state = ChainState(cfg)
        self.fast_energy = FastEnergyComputer(cfg)
        self.gen = cfg.get_torch_gen()
        self.stats = SimulationStats()
        self.snapshot_collector = snapshot_collector
        self.rng = np.random.RandomState(cfg.seed)

        self.dynamic_accum = DynamicAccumulator(cfg, self.ns)
        self.sweep_tracker = SweepTracker(cfg, self.ns)

        self.final_R2 = None
        self.final_Rg2 = None

        self._pos_np = None

    def initialize(self) -> None:
        self.state.initialize_random_walk(self.gen)
        self.state.wrap_all()
        self._pos_np = self.state.positions.cpu().numpy().copy()
        print(f"  Initialized {self.cfg.n_chains} chains of N={self.cfg.N} "
              f"(box={self.cfg.box_size:.1f} A, fast mode)", flush=True)

    def _sync_torch_from_numpy(self) -> None:
        self.state.positions.copy_(torch.from_numpy(self._pos_np))

    def run_equilibration(self) -> None:
        cfg = self.cfg
        total = cfg.eq_sweeps
        seg_info = self.state.segments

        pbar = tqdm(range(total), desc="  Eq", unit="sw",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        box = cfg.box_size
        inv_box_eq = 1.0 / box
        half_box_eq = cfg.half_box
        for sweep in pbar:
            FastSweep.perform_sweep(self._pos_np, seg_info, self.fast_energy,
                                    cfg, self.stats, self.rng)

            if cfg.n_small_steps > 0 and (sweep + 1) % cfg.n_small_steps == 0:
                drift = rescale_bonds_numpy(
                    self._pos_np, cfg.l0, box, half_box_eq, inv_box_eq)
                self.stats.rescale_count += 1
                self.stats.rescale_last_drift = drift
                if drift > self.stats.rescale_max_drift:
                    self.stats.rescale_max_drift = drift

            if sweep % max(1, total // 50) == 0:
                self._sync_torch_from_numpy()
                self.sweep_tracker.record(self.state, sweep, "equilibration")

            sc = self.snapshot_collector
            if sc is not None and getattr(sc, "should_capture", lambda *_: True)(sweep, "eq"):
                self._sync_torch_from_numpy()
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

        cum_seg_disp_np = np.zeros_like(self._pos_np)
        box = cfg.box_size
        inv_box = 1.0 / box

        pbar = tqdm(range(total), desc="  Prod", unit="sw",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        half_box_prod = cfg.half_box
        for sweep in pbar:
            pos_before = self._pos_np.copy()

            FastSweep.perform_sweep(self._pos_np, seg_info, self.fast_energy,
                                    cfg, self.stats, self.rng, skip_pivot=True)

            if cfg.n_small_steps > 0 and (sweep + 1) % cfg.n_small_steps == 0:
                drift = rescale_bonds_numpy(
                    self._pos_np, cfg.l0, box, half_box_prod, inv_box)
                self.stats.rescale_count += 1
                self.stats.rescale_last_drift = drift
                if drift > self.stats.rescale_max_drift:
                    self.stats.rescale_max_drift = drift

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

            sc = self.snapshot_collector
            if sc is not None and getattr(sc, "should_capture", lambda *_: True)(cfg.eq_sweeps + sweep, "prod"):
                self._sync_torch_from_numpy()
                sc.capture(self.state.positions, cfg.eq_sweeps + sweep, "prod")

            pbar.set_postfix_str(self.stats.report(), refresh=False)

        pbar.close()
        dt = pbar.format_dict.get('elapsed', 0)

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

    def run(self) -> dict:
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
