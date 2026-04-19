"""SimulationDefaults — dataclass loaded from configs/simulation.toml."""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import tomllib


@dataclass(frozen=True)
class SimulationDefaults:
    seed: int
    residues_per_segment: int
    max_angle_hinge: float
    max_angle_pivot: float
    sample_interval: int
    move_size: int
    n_small_steps: int
    rand_pool_initial_size: int
    rand_pool_expansion_size: int
    bond_tolerance: float
    validation_chain_lengths: List[int]
    validation_phi_values: List[float]

    @classmethod
    def from_toml(cls, path: Path) -> "SimulationDefaults":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        d = data["defaults"]
        rp = data["rand_pool"]
        b = data["bond"]
        vm = data["validation_matrix"]
        return cls(
            seed=d["seed"],
            residues_per_segment=d["residues_per_segment"],
            max_angle_hinge=d["max_angle_hinge"],
            max_angle_pivot=d["max_angle_pivot"],
            sample_interval=d["sample_interval"],
            move_size=d["move_size"],
            n_small_steps=d.get("n_small_steps", 100),
            rand_pool_initial_size=rp["initial_size"],
            rand_pool_expansion_size=rp["expansion_size"],
            bond_tolerance=b["tolerance"],
            validation_chain_lengths=list(vm["chain_lengths"]),
            validation_phi_values=list(vm["phi_values"]),
        )
