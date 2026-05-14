"""SimConfig — flat dataclass holding every run-time parameter."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SimConfig:
    N: int = 25
    n_chains: int = 4
    phi: float = 0.10

    eq_sweeps: int = 100
    prod_sweeps: int = 100

    sigma: float = 3.8
    l0_factor: float = 1.0  # Calpha-Calpha bond length = l0_factor * sigma  (l0 = sigma)
    d_ca_sg: float = 3.8
    kBT: float = 8.314462e-3 * 300.0

    repulsive_energy: float = 1e6
    contact_energy: float = 0.0
    r_rep_factor: float = 1.0
    r_max_factor: float = 2.0

    min_seq_caca: int = 2
    min_seq_casg: int = 1
    min_seq_sgsg: int = 1

    residues_per_segment: int = 8
    max_angle_hinge: float = math.pi / 2.0
    n_small_steps: int = 100

    init_method: str = "random_saw"
    seed: int = 42

    sample_interval: int = 5
    output_dir: str = "./_smoke"
    cap_inner_hinge: bool = False
    cap_tail: bool = False

    bond_stretch_warn: float = 0.02
    bond_stretch_raise: float = 0.10

    box_size: float = 0.0
    l0: float = 0.0
    r_rep: float = 0.0
    r_max: float = 0.0

    def __post_init__(self):
        self.l0 = self.l0_factor * self.sigma
        self.r_rep = self.r_rep_factor * self.sigma
        self.r_max = self.r_max_factor * self.sigma
        if self.box_size <= 0.0:
            n_beads = self.n_chains * self.N * 2
            bead_vol = (math.pi / 6.0) * self.sigma ** 3
            total_vol = n_beads * bead_vol / max(self.phi, 1e-12)
            self.box_size = total_vol ** (1.0 / 3.0)

    @property
    def n_segments_per_chain(self) -> int:
        return max(1, math.ceil(self.N / self.residues_per_segment))

    @property
    def total_segments(self) -> int:
        return self.n_chains * self.n_segments_per_chain

    @property
    def half_box(self) -> float:
        return self.box_size / 2.0

    @property
    def r_rep_sq(self) -> float:
        return self.r_rep ** 2

    @property
    def r_max_sq(self) -> float:
        return self.r_max ** 2
