"""BenchmarkMatrix — dataclass loaded from configs/benchmark.toml."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import tomllib


@dataclass(frozen=True)
class BenchmarkMatrix:
    chain_lengths: List[int]
    phi_values: List[float]
    algorithms: List[str]
    devices: List[str]
    sweep_schedule: Dict[int, Tuple[int, int]]
    timing_sweep_schedule: Dict[int, Tuple[int, int]]
    cell_defaults: Dict[str, object]
    pe1: Dict[str, object]
    pe2: Dict[str, object]
    pe3: Dict[str, object]

    @classmethod
    def from_toml(cls, path: Path) -> "BenchmarkMatrix":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        m = data["matrix"]
        ss = {int(k): (v["eq"], v["prod"]) for k, v in data["sweep_schedule"].items()}
        ts = {int(k): (v["eq"], v["prod"]) for k, v in data["timing_sweep_schedule"].items()}
        return cls(
            chain_lengths=list(m["chain_lengths"]),
            phi_values=list(m["phi_values"]),
            algorithms=list(m["algorithms"]),
            devices=list(m["devices"]),
            sweep_schedule=ss,
            timing_sweep_schedule=ts,
            cell_defaults=dict(data["cell_defaults"]),
            pe1=dict(data["pe1"]),
            pe2=dict(data["pe2"]),
            pe3=dict(data.get("pe3", {})),
        )
