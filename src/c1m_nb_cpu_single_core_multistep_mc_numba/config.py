"""SimConfig — flat dataclass holding every run-time parameter.

Self-contained: no imports from src/libs. Defaults track physics.toml
(sigma=3.8 A, l0=1.5*sigma=5.7 A, kBT for T=300 K) but the SURPASS-alpha
geometry adds an SG bead per residue at distance d_ca_sg from its parent
Calpha.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # System size
    N: int = 25                            # residues per chain (= number of Calpha = number of SG)
    n_chains: int = 4
    phi: float = 0.10                      # volume fraction (Calpha + SG combined)

    # Sweep counts
    eq_sweeps: int = 100
    prod_sweeps: int = 100

    # Physics (reduced units: sigma is the bead diameter)
    sigma: float = 3.8                     # Calpha bead diameter (A)
    l0_factor: float = 1.0                 # Calpha-Calpha bond length = l0_factor * sigma  (l0 = sigma)
    d_ca_sg: float = 3.8                   # Calpha-SG bond length (A); rigid
    kBT: float = 8.314462e-3 * 300.0       # kJ/mol at T=300K

    repulsive_energy: float = 1e6          # excluded-volume penalty
    contact_energy: float = 0.0            # athermal default
    r_rep_factor: float = 1.0              # r_rep = r_rep_factor * sigma
    r_max_factor: float = 2.0              # r_max = r_max_factor * sigma

    # Sequence-separation exclusions (skip pair if |i-j| < this for same chain)
    min_seq_caca: int = 2
    min_seq_casg: int = 1
    min_seq_sgsg: int = 1

    # MC
    residues_per_segment: int = 8
    max_angle_hinge: float = math.pi / 2.0
    batch_size: int = 256
    n_small_steps: int = 100               # bond rescale every n sweeps

    # Init
    init_method: str = "random_saw"        # "random_saw" or "serpentine"
    seed: int = 42

    # Run-control / IO
    sample_interval: int = 5
    output_dir: str = "./_smoke"
    snapshot_pdb: bool = True
    cap_inner_hinge: bool = False
    cap_tail: bool = False

    bond_stretch_warn: float = 0.02
    bond_stretch_raise: float = 0.10

    # Computed at construction (kept here to make passing into kernels uniform).
    box_size: float = 0.0
    l0: float = 0.0
    r_rep: float = 0.0
    r_max: float = 0.0

    def __post_init__(self):
        self.l0 = self.l0_factor * self.sigma
        self.r_rep = self.r_rep_factor * self.sigma
        self.r_max = self.r_max_factor * self.sigma
        if self.box_size <= 0.0:
            self.box_size = self._compute_box_size()

    def _compute_box_size(self) -> float:
        # Each residue contributes one Calpha + one SG to the volume budget.
        n_beads = self.n_chains * self.N * 2
        bead_vol = (math.pi / 6.0) * self.sigma ** 3
        total_vol = n_beads * bead_vol / max(self.phi, 1e-12)
        return total_vol ** (1.0 / 3.0)

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
