"""ConfigLoader — aggregates PhysicalConstants, BenchmarkMatrix,
SimulationDefaults, ExecutionTuning from their respective TOML files.

Default configs dir is `<package root>/configs` resolved relative to
this file, so the apps can instantiate `ConfigLoader()` with no args.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rouse_model_python.src.libs.config.physical_constants import PhysicalConstants
from rouse_model_python.src.libs.config.benchmark_matrix import BenchmarkMatrix
from rouse_model_python.src.libs.config.simulation_defaults import SimulationDefaults
from rouse_model_python.src.libs.config.execution_tuning import ExecutionTuning


@dataclass(frozen=True)
class ConfigLoader:
    physics: PhysicalConstants
    benchmark: BenchmarkMatrix
    simulation: SimulationDefaults
    execution: ExecutionTuning

    @classmethod
    def default_configs_dir(cls) -> Path:
        # <repo>/rouse_model_python/libs/config/config_loader.py
        # -> <repo>/rouse_model_python/configs
        return Path(__file__).resolve().parents[2] / "configs"

    @classmethod
    def load(cls, configs_dir: Optional[Path] = None) -> "ConfigLoader":
        root = Path(configs_dir) if configs_dir is not None else cls.default_configs_dir()
        missing = [p for p in (
            "physics.toml", "benchmark.toml", "simulation.toml", "execution.toml"
        ) if not (root / p).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing TOML file(s) under {root}: {missing}"
            )
        return cls(
            physics=PhysicalConstants.from_toml(root / "physics.toml"),
            benchmark=BenchmarkMatrix.from_toml(root / "benchmark.toml"),
            simulation=SimulationDefaults.from_toml(root / "simulation.toml"),
            execution=ExecutionTuning.from_toml(root / "execution.toml"),
        )
