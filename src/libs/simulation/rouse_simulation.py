"""RouseSimulation — orchestrator for init + equilibration + production."""

import logging

import torch
from tqdm import tqdm

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.chain.chain_state import ChainState
from rouse_model_python.src.libs.chain.bond_rescaler import rescale_bonds_torch
from rouse_model_python.src.libs.number_space.number_space import NumberSpace
from rouse_model_python.src.libs.energy.energy_computer import EnergyComputer
from rouse_model_python.src.libs.observables.static_observables import StaticObservables
from rouse_model_python.src.libs.observables.dynamic_accumulator import DynamicAccumulator
from rouse_model_python.src.libs.observables.sweep_tracker import SweepTracker
from rouse_model_python.src.libs.simulation.simulation_stats import SimulationStats
from rouse_model_python.src.libs.simulation.sweep_dispatcher import SweepDispatcher


class RouseSimulation:
    _logger = logging.getLogger(__name__)

    def __init__(self, cfg: SimulationConfig, snapshot_collector=None):
        self.cfg = cfg
        self.ns = NumberSpace.from_config(cfg)
        self.state = ChainState(cfg)
        self.energy_comp = EnergyComputer(cfg, self.ns)
        self.gen = cfg.get_torch_gen()
        self.stats = SimulationStats()
        self.snapshot_collector = snapshot_collector
        self.dynamic_accum = DynamicAccumulator(cfg, self.ns)
        self.sweep_tracker = SweepTracker(cfg, self.ns)
        self.final_R2 = None
        self.final_Rg2 = None

    def initialize(self):
        self.state.initialize_random_walk(self.gen)
        self.state.wrap_all()
        print(f"  Initialized {self.cfg.n_chains} chains of N={self.cfg.N} "
              f"(box={self.cfg.box_size:.1f} A)", flush=True)

    def run_equilibration(self):
        cfg = self.cfg
        total = cfg.eq_sweeps
        sweep_fn, sweep_path = SweepDispatcher.resolve(cfg)
        self._logger.info("[BENCH-B2] Algorithm=%s SweepPath=%s Phase=equilibration",
                    getattr(cfg, "algorithm", "multistep"), sweep_path)
        self._logger.info(
            "[CHECKLIST-C4] Phase=eq MoveTypes=hinge,n_tail,c_tail,pivot Count=4")
        pbar = tqdm(range(total), desc="  Eq", unit="sw",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        for sweep in pbar:
            sweep_fn(self.state, self.energy_comp, self.gen, cfg, self.stats, skip_pivot=False)
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

    def run_production(self):
        cfg = self.cfg
        ns = self.ns
        total = cfg.prod_sweeps
        sample_interval = cfg.sample_interval
        sweep_fn, sweep_path = SweepDispatcher.resolve(cfg)
        self._logger.info("[BENCH-B2] Algorithm=%s SweepPath=%s Phase=production",
                    getattr(cfg, "algorithm", "multistep"), sweep_path)
        self._logger.info(
            "[CHECKLIST-C4] Phase=prod MoveTypes=hinge,n_tail,c_tail Count=3")
        self._logger.info(
            "[CHECKLIST-C32] FirstProductionSweep=%d EquilibrationSweeps=%d",
            cfg.eq_sweeps + 1, cfg.eq_sweeps)
        pivot_before = self.stats.pivot_attempted
        pbar = tqdm(range(total), desc="  Prod", unit="sw",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        for sweep in pbar:
            sweep_fn(self.state, self.energy_comp, self.gen, cfg, self.stats, skip_pivot=True)
            if cfg.n_small_steps > 0 and (sweep + 1) % cfg.n_small_steps == 0:
                drift = rescale_bonds_torch(
                    self.state.positions, cfg.l0, cfg.box_size, cfg.half_box)
                self.stats.rescale_count += 1
                self.stats.rescale_last_drift = drift
                if drift > self.stats.rescale_max_drift:
                    self.stats.rescale_max_drift = drift
            if sweep % max(1, total // 50) == 0:
                self.sweep_tracker.record(self.state, cfg.eq_sweeps + sweep, "production")
            if sweep % sample_interval == 0:
                self.dynamic_accum.record_snapshot(self.state, sweep)
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
        print(f"  <R2>={mean_R2:.1f}, <Rg2>={mean_Rg2:.1f}, R2/Rg2={ratio:.2f}")
        pivot_in_prod = self.stats.pivot_attempted - pivot_before
        self._logger.info(
            "[CHECKLIST-C26] HingeAccept=%.6f NTailAccept=%.6f CTailAccept=%.6f "
            "PivotAccept=%.6f PivotAttemptedInProd=%d",
            (self.stats.hinge_accepted / self.stats.hinge_attempted) if self.stats.hinge_attempted > 0 else 0.0,
            (self.stats.n_tail_accepted / self.stats.n_tail_attempted) if self.stats.n_tail_attempted > 0 else 0.0,
            (self.stats.c_tail_accepted / self.stats.c_tail_attempted) if self.stats.c_tail_attempted > 0 else 0.0,
            (self.stats.pivot_accepted / self.stats.pivot_attempted) if self.stats.pivot_attempted > 0 else 0.0,
            pivot_in_prod)

    def run(self):
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
