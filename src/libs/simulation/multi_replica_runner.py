"""MultiReplicaRunner — run R RouseSimulation replicas concurrently on separate CUDA streams.

Each replica runs an independent Migacz multistep MC chain at the SAME
temperature; there are NO replica-exchange moves. The purpose is to
fill the idle 70% of GPU time that a single-replica run leaves on the
table at small n_chains — stream-level kernel overlap lets the
scheduler interleave work from R chains, boosting GpuUtilMean from
~0.30 toward 0.8+.

Benchmark framing: the useful metric is throughput, not latency. R
replicas running in parallel do NOT make any single chain finish
faster — they produce R chains' worth of samples in roughly the same
wall time that one replica alone would take. Call the normalised
metric `per_sweep_s_effective = total_wall / (R * total_sweeps)` when
comparing to a CPU single-replica baseline.

TODO: replica-exchange moves (parallel tempering proper) — each
replica at a different temperature, periodic Metropolis swap attempts
on energy. Currently out of scope by explicit request; the runner is
structured so that adding an exchange step between sweep calls is a
localized addition.
"""

import dataclasses
import logging
import torch

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.simulation.rouse_simulation import RouseSimulation
from rouse_model_python.src.libs.simulation.sweep_dispatcher import SweepDispatcher


class MultiReplicaRunner:
    _logger = logging.getLogger(__name__)

    def __init__(self, cfg: SimulationConfig, n_replicas: int,
                 seed_base: int = None, snapshot_collector=None):
        if n_replicas < 1:
            raise ValueError(f"n_replicas must be >= 1, got {n_replicas}")
        self.cfg = cfg
        self.n_replicas = n_replicas
        self.seed_base = seed_base if seed_base is not None else cfg.seed
        device = cfg.get_torch_device()
        use_streams = (device.type == "cuda" and n_replicas > 1)
        self.replicas = [self._make_replica(i, snapshot_collector if i == 0 else None)
                         for i in range(n_replicas)]
        self.streams = ([torch.cuda.Stream() for _ in range(n_replicas)]
                        if use_streams else [None] * n_replicas)
        self._use_streams = use_streams
        self._sweep_fn, self._sweep_path = SweepDispatcher.resolve(cfg)

    def _make_replica(self, i, snapshot_collector):
        cfg_i = dataclasses.replace(self.cfg, seed=self.seed_base + i)
        sim = RouseSimulation(cfg_i, snapshot_collector=snapshot_collector)
        return sim

    def initialize(self):
        for sim in self.replicas:
            sim.initialize()

    def _run_phase(self, n_sweeps: int, skip_pivot: bool):
        sweep_fn = self._sweep_fn
        if self._use_streams:
            for _ in range(n_sweeps):
                for sim, stream in zip(self.replicas, self.streams):
                    with torch.cuda.stream(stream):
                        sweep_fn(sim.state, sim.energy_comp, sim.gen,
                                 sim.cfg, sim.stats, skip_pivot=skip_pivot)
            # One drain at end of phase — per-sweep sync would serialize the streams.
            torch.cuda.synchronize()
        else:
            for _ in range(n_sweeps):
                for sim in self.replicas:
                    sweep_fn(sim.state, sim.energy_comp, sim.gen,
                             sim.cfg, sim.stats, skip_pivot=skip_pivot)

    def run_equilibration(self):
        self._logger.info(
            "[BENCH-MR] Phase=eq Replicas=%d SweepPath=%s Streams=%s",
            self.n_replicas, self._sweep_path, self._use_streams,
        )
        self._run_phase(self.cfg.eq_sweeps, skip_pivot=False)

    def run_production(self):
        self._logger.info(
            "[BENCH-MR] Phase=prod Replicas=%d SweepPath=%s Streams=%s",
            self.n_replicas, self._sweep_path, self._use_streams,
        )
        self._run_phase(self.cfg.prod_sweeps, skip_pivot=True)

    def aggregate_final_observables(self):
        """Return mean R^2 and mean Rg^2 averaged across replicas."""
        from rouse_model_python.src.libs.observables.static_observables import StaticObservables
        r2_sum = 0.0
        rg2_sum = 0.0
        count = 0
        for sim in self.replicas:
            r2 = StaticObservables.compute_R2(sim.state.positions, sim.ns)
            rg2 = StaticObservables.compute_Rg2(sim.state.positions, sim.ns)
            r2_sum += float(r2.mean().item())
            rg2_sum += float(rg2.mean().item())
            count += 1
        return r2_sum / count, rg2_sum / count

    def aggregate_stats(self):
        """Return a dict of summed-across-replica acceptance counts + rates."""
        agg = {
            "hinge_attempted": 0, "hinge_accepted": 0,
            "n_tail_attempted": 0, "n_tail_accepted": 0,
            "c_tail_attempted": 0, "c_tail_accepted": 0,
            "pivot_attempted": 0, "pivot_accepted": 0,
        }
        for sim in self.replicas:
            s = sim.stats
            agg["hinge_attempted"] += s.hinge_attempted
            agg["hinge_accepted"] += s.hinge_accepted
            agg["n_tail_attempted"] += s.n_tail_attempted
            agg["n_tail_accepted"] += s.n_tail_accepted
            agg["c_tail_attempted"] += s.c_tail_attempted
            agg["c_tail_accepted"] += s.c_tail_accepted
            agg["pivot_attempted"] += s.pivot_attempted
            agg["pivot_accepted"] += s.pivot_accepted
        def _rate(num, denom):
            return float(num) / float(denom) if denom > 0 else 0.0
        agg["hinge_rate"] = _rate(agg["hinge_accepted"], agg["hinge_attempted"])
        agg["tail_rate"] = _rate(
            agg["n_tail_accepted"] + agg["c_tail_accepted"],
            agg["n_tail_attempted"] + agg["c_tail_attempted"])
        agg["pivot_rate"] = _rate(agg["pivot_accepted"], agg["pivot_attempted"])
        return agg
