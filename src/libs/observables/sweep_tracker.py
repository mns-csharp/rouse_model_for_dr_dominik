"""SweepTracker — per-sweep running averages of R² and Rg²."""

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.chain.chain_state import ChainState
from rouse_model_python.src.libs.number_space.number_space import NumberSpace
from rouse_model_python.src.libs.observables.static_observables import StaticObservables


class SweepTracker:
    def __init__(self, cfg: SimulationConfig, ns: NumberSpace):
        self.cfg = cfg
        self.ns = ns
        self.records = []

    def record(self, state: ChainState, sweep: int, phase: str):
        positions = state.positions
        ns = self.ns
        R2 = StaticObservables.compute_R2(positions, ns)
        Rg2 = StaticObservables.compute_Rg2(positions, ns)
        mean_R2 = R2.mean().item()
        mean_Rg2 = Rg2.mean().item()
        ratio = mean_R2 / mean_Rg2 if mean_Rg2 > 1e-30 else 0.0
        self.records.append((sweep, phase, mean_R2, mean_Rg2, ratio))
