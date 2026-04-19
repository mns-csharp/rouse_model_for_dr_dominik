"""SimulationConfig — per-run configuration dataclass.

Holds everything the simulation needs at run-time: system size, sweep
counts, device, physical parameters (sourced from PhysicalConstants),
move parameters, algorithm choice, batching flag.
"""

from dataclasses import dataclass
from typing import Optional
import math
import torch

from rouse_model_python.src.libs.config.physical_constants import PhysicalConstants
from rouse_model_python.src.libs.config.simulation_defaults import SimulationDefaults
from rouse_model_python.src.libs.config.config_helpers import ConfigHelpers


@dataclass
class SimulationConfig:
    # System size
    N: int
    n_chains: int
    eq_sweeps: int
    prod_sweeps: int
    box_size: float

    # Seeding / device
    seed: int = 42
    device: str = ""                    # REQUIRED: must be set explicitly
    dtype: torch.dtype = torch.float64

    # Physical parameters (defaults mirror physics.toml for backwards compat;
    # overridden at construction from a PhysicalConstants instance)
    sigma: float = 3.8
    l0: float = 3.8
    kBT: float = 8.314462e-3 * 300.0
    repulsive_energy: float = 1e6
    contact_energy: float = 0.0
    r_rep: float = 3.8
    r_min: float = 3.8
    r_max: float = 7.6
    target_phi: float = 0.035

    # MC parameters
    residues_per_segment: int = 20
    max_angle_hinge: float = math.pi / 2.0
    max_angle_pivot: float = math.pi
    sample_interval: int = 5
    n_small_steps: int = 100

    # Execution mode
    use_batched_mode: bool = False
    algorithm: str = "multistep"
    batch_size: int = 100
    use_gpu_energy_path: bool = False
    accept_on_gpu: bool = False
    delta_e_mem_budget_gib: float = 2.0
    pivot_on_gpu: bool = False
    pivot_pad_strategy: str = "tight"

    @classmethod
    def from_defaults(cls, N: int, n_chains: int, eq_sweeps: int,
                      prod_sweeps: int, box_size: float, device: str,
                      physics: PhysicalConstants,
                      defaults: SimulationDefaults,
                      target_phi: float = 0.035,
                      algorithm: str = "multistep") -> "SimulationConfig":
        if not device:
            raise ValueError("device is required (no default). Pass 'cpu' or 'cuda'.")
        return cls(
            N=N, n_chains=n_chains, eq_sweeps=eq_sweeps, prod_sweeps=prod_sweeps,
            box_size=box_size, seed=defaults.seed, device=device,
            sigma=physics.SIGMA, l0=physics.L0, kBT=physics.KBT,
            repulsive_energy=physics.REPULSIVE_ENERGY,
            contact_energy=physics.CONTACT_ENERGY,
            r_rep=physics.R_REP, r_min=physics.R_MIN, r_max=physics.R_MAX,
            target_phi=target_phi,
            residues_per_segment=defaults.residues_per_segment,
            max_angle_hinge=defaults.max_angle_hinge,
            max_angle_pivot=defaults.max_angle_pivot,
            sample_interval=defaults.sample_interval,
            n_small_steps=defaults.n_small_steps,
            algorithm=algorithm,
        )

    @classmethod
    def for_state_point(cls, N: int, phi: float, device: str,
                        physics: PhysicalConstants,
                        defaults: SimulationDefaults,
                        eq_sweeps: Optional[int] = None,
                        prod_sweeps: Optional[int] = None,
                        n_chains: Optional[int] = None) -> "SimulationConfig":
        if not device:
            raise ValueError("device is required (no default). Pass 'cpu' or 'cuda'.")
        nc = n_chains if n_chains is not None else ConfigHelpers.compute_n_chains(N, phi)
        bs = ConfigHelpers.compute_box_size(N, nc, phi, physics)
        si = max(1, min(defaults.sample_interval, N // 10))
        cfg = cls.from_defaults(
            N=N, n_chains=nc,
            eq_sweeps=eq_sweeps if eq_sweeps is not None else 500,
            prod_sweeps=prod_sweeps if prod_sweeps is not None else 500,
            box_size=bs, device=device,
            physics=physics, defaults=defaults, target_phi=phi,
        )
        cfg.sample_interval = si
        return cfg

    @property
    def n_segments_per_chain(self) -> int:
        return math.ceil(self.N / self.residues_per_segment)

    @property
    def total_segments(self) -> int:
        return self.n_chains * self.n_segments_per_chain

    @property
    def total_beads(self) -> int:
        return self.n_chains * self.N

    @property
    def half_box(self) -> float:
        return self.box_size / 2.0

    @property
    def r_max_sq(self) -> float:
        return self.r_max ** 2

    @property
    def r_rep_sq(self) -> float:
        return self.r_rep ** 2

    def get_torch_device(self) -> torch.device:
        d = self.device
        if not d:
            raise ValueError("SimulationConfig.device is not set.")
        if d == "gpu":
            d = "cuda"
        return torch.device(d)

    def get_torch_gen(self) -> torch.Generator:
        gen = torch.Generator(device=self.get_torch_device())
        gen.manual_seed(self.seed)
        return gen
