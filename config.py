"""
SimulationConfig: all physical and simulation parameters for Rouse-model MC.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import math
import torch


# Physical constants
SIGMA = 3.8            # Bead diameter (Angstrom)
L0 = 1.5 * SIGMA      # Equilibrium bond length = 5.7 A
KB = 8.314462e-3       # Boltzmann constant (kJ/(mol*K))
TEMPERATURE = 300.0    # Kelvin
KBT = KB * TEMPERATURE # ~2.494 kJ/mol

# Energy parameters (athermal excluded-volume)
REPULSIVE_ENERGY = 1e6  # kJ/mol  (hard-core repulsion for r < sigma)
CONTACT_ENERGY = 0.0    # kJ/mol  (no attractive interactions)

# Cutoff radii for 3-zone kernel
R_REP = SIGMA           # 3.8 A  (repulsive boundary)
R_MIN = SIGMA           # 3.8 A  (neutral zone starts)
R_MAX = 2.0 * SIGMA     # 7.6 A  (contact zone ends)

# Volume fraction
TARGET_PHI = 0.035

# Default segment size for segmented multi-step MC
RESIDUES_PER_SEGMENT = 20

# MC move parameters
MAX_ANGLE_HINGE = math.pi / 2   # 90 degrees for hinge/tail moves
MAX_ANGLE_PIVOT = math.pi       # Full rotation range for pivot

# Random seed
SEED = 42

# Per-chain-length simulation parameters
# (N, n_chains, eq_sweeps, prod_sweeps, box_size)
CHAIN_CONFIGS: Dict[int, Tuple[int, int, int, float]] = {
    25:  (500, 2000,  5000,  269.6),
    50:  (500, 5000,  5000,  339.7),
    100: (500, 5000,  5000,  428.0),
    250: (200, 10000, 5000,  428.0),
    500: (110, 25000, 10000, 441.8),
}

# Dynamic sampling interval (sweeps between observable snapshots)
SAMPLE_INTERVAL = 20


@dataclass
class SimulationConfig:
    """Configuration for a single Rouse-model MC simulation run."""

    N: int                    # Beads per chain
    n_chains: int             # Number of chains
    eq_sweeps: int            # Equilibration sweeps
    prod_sweeps: int          # Production sweeps
    box_size: float           # Cubic box side length (Angstrom)
    seed: int = SEED
    device: str = "cpu"       # "cpu" or "cuda"
    dtype: torch.dtype = torch.float64

    # Physical parameters (fixed)
    sigma: float = SIGMA
    l0: float = L0
    kBT: float = KBT
    repulsive_energy: float = REPULSIVE_ENERGY
    contact_energy: float = CONTACT_ENERGY
    r_rep: float = R_REP
    r_min: float = R_MIN
    r_max: float = R_MAX
    target_phi: float = TARGET_PHI

    # MC parameters
    residues_per_segment: int = RESIDUES_PER_SEGMENT
    max_angle_hinge: float = MAX_ANGLE_HINGE
    max_angle_pivot: float = MAX_ANGLE_PIVOT
    sample_interval: int = SAMPLE_INTERVAL

    # Execution mode
    use_batched_mode: bool = False  # batched proposals + delta-E (works on CPU and GPU)

    @classmethod
    def for_chain_length(cls, N: int, device: str = "cpu") -> "SimulationConfig":
        """Create config for a standard chain length (25, 50, 100, 250, 500)."""
        n_chains, eq_sweeps, prod_sweeps, box_size = CHAIN_CONFIGS[N]
        return cls(
            N=N,
            n_chains=n_chains,
            eq_sweeps=eq_sweeps,
            prod_sweeps=prod_sweeps,
            box_size=box_size,
            device=device,
        )

    @property
    def n_segments_per_chain(self) -> int:
        """Number of segments per chain (ceiling division)."""
        return math.ceil(self.N / self.residues_per_segment)

    @property
    def total_segments(self) -> int:
        """Total number of segments across all chains."""
        return self.n_chains * self.n_segments_per_chain

    @property
    def total_beads(self) -> int:
        """Total number of beads in the system."""
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

    def compute_box_size(self) -> float:
        """Compute box size from volume fraction (for verification)."""
        sigma3 = self.sigma ** 3
        box_from_phi = (self.n_chains * self.N * sigma3 / self.target_phi) ** (1.0 / 3.0)
        rms_R = self.sigma * (self.N ** 0.588)
        return max(box_from_phi, 3.0 * rms_R)

    def get_torch_device(self) -> torch.device:
        d = self.device
        if d == "gpu":
            d = "cuda"
        return torch.device(d)

    def get_torch_gen(self) -> torch.Generator:
        """Create a seeded torch Generator on the configured device."""
        gen = torch.Generator(device=self.get_torch_device())
        gen.manual_seed(self.seed)
        return gen
