"""SweepDispatcher — resolves the MC algorithm (conventional vs multistep)
to a concrete sweep function for the current SimulationConfig.
"""

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.algorithms.multistep_mc import MultiStepMC
from rouse_model_python.src.libs.algorithms.conventional_mc import ConventionalMC


class SweepDispatcher:
    @staticmethod
    def resolve(cfg: SimulationConfig):
        """Return (sweep_fn, sweep_path_str)."""
        alg = getattr(cfg, "algorithm", "multistep")
        if alg == "conventional":
            return ConventionalMC.run_sweep, "conventional_mc.run_sweep"
        return MultiStepMC.perform_sweep, "multistep_mc.perform_sweep"
