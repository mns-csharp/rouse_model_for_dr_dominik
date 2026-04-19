"""DynamicAccumulator — time-lagged dynamic observables g1, gCM, gR."""

import torch

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.chain.chain_state import ChainState
from rouse_model_python.src.libs.number_space.number_space import NumberSpace
from rouse_model_python.src.libs.observables.static_observables import StaticObservables


class DynamicAccumulator:
    def __init__(self, cfg: SimulationConfig, ns: NumberSpace):
        self.cfg = cfg
        self.ns = ns
        self.device = cfg.get_torch_device()
        self.dtype = cfg.dtype
        self.n_chains = cfg.n_chains
        self.N = cfg.N
        self._prev_positions = None
        self._cum_bead_disp = None
        self.mid_refs = []
        self.cm_refs = []
        self.ee_refs = []
        self.ee_norm_sq_sum = 0.0
        self.ee_norm_count = 0
        self.g1_accum = {}
        self.gcm_accum = {}
        self.gr_accum = {}
        self._snapshot_count = 0
        self._max_refs = 500
        self._ref_modulo = 1

    def record_snapshot(self, state: ChainState, sweep: int,
                        cum_bead_disp=None):
        positions = state.positions
        ns = self.ns
        mid_idx = self.N // 2
        if cum_bead_disp is not None:
            pass
        else:
            if self._prev_positions is None:
                self._cum_bead_disp = torch.zeros_like(positions)
                self._prev_positions = positions.clone()
            else:
                delta = ns.mic_delta(self._prev_positions, positions)
                self._cum_bead_disp = self._cum_bead_disp + delta
                self._prev_positions = positions.clone()
            cum_bead_disp = self._cum_bead_disp
        cum_mid = cum_bead_disp[:, mid_idx, :].clone()
        cum_cm = cum_bead_disp.mean(dim=1).clone()
        ee_vec = StaticObservables.compute_end_to_end_vector(positions, ns)
        ee_sq = (ee_vec * ee_vec).sum(dim=1)
        self.ee_norm_sq_sum += ee_sq.sum().item()
        self.ee_norm_count += self.n_chains
        for ref_sweep, ref_mid in self.mid_refs:
            lag = sweep - ref_sweep
            d = cum_mid - ref_mid
            msd = (d * d).sum(dim=1).mean().item()
            if lag not in self.g1_accum:
                self.g1_accum[lag] = [0.0, 0]
            self.g1_accum[lag][0] += msd
            self.g1_accum[lag][1] += 1
        for ref_sweep, ref_cm in self.cm_refs:
            lag = sweep - ref_sweep
            d = cum_cm - ref_cm
            msd = (d * d).sum(dim=1).mean().item()
            if lag not in self.gcm_accum:
                self.gcm_accum[lag] = [0.0, 0]
            self.gcm_accum[lag][0] += msd
            self.gcm_accum[lag][1] += 1
        for ref_sweep, ref_ee in self.ee_refs:
            lag = sweep - ref_sweep
            dot = (ref_ee * ee_vec).sum(dim=1).mean().item()
            if lag not in self.gr_accum:
                self.gr_accum[lag] = [0.0, 0]
            self.gr_accum[lag][0] += dot
            self.gr_accum[lag][1] += 1
        self._snapshot_count += 1
        total_expected = self.cfg.prod_sweeps // self.cfg.sample_interval
        if total_expected > self._max_refs:
            self._ref_modulo = max(1, total_expected // self._max_refs)
        if self._snapshot_count % self._ref_modulo == 0 or self._snapshot_count <= 10:
            self.mid_refs.append((sweep, cum_mid))
            self.cm_refs.append((sweep, cum_cm))
            self.ee_refs.append((sweep, ee_vec.clone()))

    def update_prev_positions(self, state: ChainState):
        if self._prev_positions is not None:
            self._prev_positions = state.positions.clone()

    def get_g1(self) -> dict:
        result = {}
        for lag, (s, c) in sorted(self.g1_accum.items()):
            if c > 0:
                result[lag] = s / c
        return result

    def get_gcm(self) -> dict:
        result = {}
        for lag, (s, c) in sorted(self.gcm_accum.items()):
            if c > 0:
                result[lag] = s / c
        return result

    def get_gr(self) -> dict:
        if self.ee_norm_count == 0:
            return {}
        norm_sq = self.ee_norm_sq_sum / self.ee_norm_count
        if norm_sq < 1e-30:
            return {}
        result = {}
        for lag, (s, c) in sorted(self.gr_accum.items()):
            if c > 0:
                result[lag] = (s / c) / norm_sq
        return result
