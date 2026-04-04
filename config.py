"""
SimulationConfig: all physical and simulation parameters for Rouse-model MC.

System is ATHERMAL: excluded-volume energy E is effectively infinite
(RepulsiveEnergy = 1e6 >> kBT). Any overlapping move is always rejected.
Temperature is vestigial in the Metropolis criterion — it does not affect
the accept/reject outcome because exp(-1e6/kBT) = 0 for any finite T.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import torch


# Physical constants
SIGMA = 3.8            # Bead diameter / Calpha bead diameter d0 (Angstrom)
L0 = 1.5 * SIGMA      # Equilibrium bond length = 5.7 A (l0/d0 = 1.5, chosen by M.N. Saqib)
KB = 8.314462e-3       # Boltzmann constant (kJ/(mol*K))
# Temperature is vestigial: the system is athermal (E=0 or 1e6).
# kBT appears in Metropolis criterion but exp(-1e6/kBT) = 0 always rejects overlaps.
TEMPERATURE = 300.0    # Kelvin (vestigial — does not affect physics)
KBT = KB * TEMPERATURE # ~2.494 kJ/mol (vestigial)

# Energy parameters (athermal excluded-volume)
REPULSIVE_ENERGY = 1e6  # kJ/mol  (hard-core repulsion for r < sigma, effectively infinite)
CONTACT_ENERGY = 0.0    # kJ/mol  (no attractive interactions — purely repulsive / athermal)

# Cutoff radii for 3-zone kernel
R_REP = SIGMA           # 3.8 A  (repulsive boundary)
R_MIN = SIGMA           # 3.8 A  (neutral zone starts)
R_MAX = 2.0 * SIGMA     # 7.6 A  (contact zone ends)

# Density progression: 6 volume fraction levels (dilute → dense)
PHI_VALUES: List[float] = [0.001, 0.01, 0.05, 0.10, 0.20, 0.30]

# Chain lengths to simulate
CHAIN_LENGTHS: List[int] = [25, 50, 100, 250, 500]

# Default segment size for segmented multi-step MC
RESIDUES_PER_SEGMENT = 20

# MC move parameters
MAX_ANGLE_HINGE = math.pi / 2   # 90 degrees for hinge/tail moves
MAX_ANGLE_PIVOT = math.pi       # Full rotation range for pivot

# Random seed
SEED = 42

# Legacy per-chain-length configs (single phi=0.035, kept for backward compat)
# (n_chains, eq_sweeps, prod_sweeps, box_size)
CHAIN_CONFIGS: Dict[int, Tuple[int, int, int, float]] = {
    25:  (50,  500, 500, 238.0),
    50:  (50,  500, 500, 238.0),
    100: (50,  500, 500, 238.0),
    250: (30,  500, 500, 293.0),
    500: (20,  500, 500, 440.5),
}

# Dynamic sampling interval (sweeps between observable snapshots)
SAMPLE_INTERVAL = 5


def compute_n_chains(N: int, phi: float) -> int:
    """Choose chain count balancing statistical quality vs computation cost.

    At low phi (dilute), fewer chains suffice. At high phi (dense), more chains
    are needed to fill the box. Long chains (N>=250) use fewer chains to keep
    memory bounded.
    """
    if phi >= 0.10:
        base = 10
    elif phi >= 0.05:
        base = 8
    elif phi >= 0.01:
        base = 5
    else:
        base = 3
    # Reduce for long chains
    if N >= 500:
        base = max(2, base // 3)
    elif N >= 250:
        base = max(3, base // 2)
    return base


def compute_box_size(N: int, n_chains: int, phi: float) -> float:
    """Compute cubic box side length from volume fraction.

    L_box = (n_chains * N * sigma^3 / phi)^(1/3)

    A floor of 3 * sigma * N^0.588 (3x RMS end-to-end distance) is applied
    only for dilute systems (phi < 0.01) where PBC self-interaction matters.
    In dense systems the box is full of chains and this floor is irrelevant.
    """
    sigma3 = SIGMA ** 3
    box_from_phi = (n_chains * N * sigma3 / phi) ** (1.0 / 3.0)
    if phi < 0.01:
        rms_R = SIGMA * (N ** 0.588)
        return max(box_from_phi, 3.0 * rms_R)
    return box_from_phi


def format_phi(phi: float) -> str:
    """Format phi for directory/file names: 0.001, 0.01, 0.05, 0.10, 0.20, 0.30."""
    if phi < 0.01:
        return f"{phi:.3f}"
    else:
        return f"{phi:.2f}"


@dataclass
class SimulationConfig:
    """Configuration for a single Rouse-model MC simulation run."""

    N: int                    # Beads per chain
    n_chains: int             # Number of chains
    eq_sweeps: int            # Equilibration sweeps
    prod_sweeps: int          # Production sweeps
    box_size: float           # Cubic box side length (Angstrom)
    seed: int = SEED
    device: str = ""          # REQUIRED: must be set explicitly (no default)
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
    target_phi: float = 0.035  # default; overridden by for_state_point()

    # MC parameters
    residues_per_segment: int = RESIDUES_PER_SEGMENT
    max_angle_hinge: float = MAX_ANGLE_HINGE
    max_angle_pivot: float = MAX_ANGLE_PIVOT
    sample_interval: int = SAMPLE_INTERVAL

    # Execution mode
    use_batched_mode: bool = False  # batched proposals + delta-E (works on CPU and GPU)

    @classmethod
    def for_chain_length(cls, N: int, device: str) -> "SimulationConfig":
        """Create config for a standard chain length (legacy single-phi mode).

        Args:
            N: chain length (must be in CHAIN_CONFIGS)
            device: torch device string -- REQUIRED, no default
        """
        if not device:
            raise ValueError("device is required (no default). Pass 'cpu' or 'cuda'.")
        n_chains, eq_sweeps, prod_sweeps, box_size = CHAIN_CONFIGS[N]
        return cls(
            N=N,
            n_chains=n_chains,
            eq_sweeps=eq_sweeps,
            prod_sweeps=prod_sweeps,
            box_size=box_size,
            device=device,
        )

    @classmethod
    def for_state_point(cls, N: int, phi: float, device: str,
                        eq_sweeps: int = 500, prod_sweeps: int = 500
                        ) -> "SimulationConfig":
        """Create config for a specific (N, phi) state point.

        Computes chain count and box size from the volume fraction formula:
            L_box = sigma * (n_chains * N / phi)^(1/3)

        Args:
            N: beads per chain
            phi: volume fraction
            device: torch device string -- REQUIRED, no default
            eq_sweeps: equilibration sweeps
            prod_sweeps: production sweeps
        """
        if not device:
            raise ValueError("device is required (no default). Pass 'cpu' or 'cuda'.")
        n_chains = compute_n_chains(N, phi)
        box_size = compute_box_size(N, n_chains, phi)
        # Adaptive sample interval: short chains relax fast, need finer sampling
        # to resolve tau_R (which is ~N^2.18).
        si = max(1, min(SAMPLE_INTERVAL, N // 10))
        return cls(
            N=N,
            n_chains=n_chains,
            eq_sweeps=eq_sweeps,
            prod_sweeps=prod_sweeps,
            box_size=box_size,
            target_phi=phi,
            device=device,
            sample_interval=si,
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
        if not d:
            raise ValueError(
                "SimulationConfig.device is not set. "
                "Device must be explicitly provided (no default)."
            )
        if d == "gpu":
            d = "cuda"
        return torch.device(d)

    def get_torch_gen(self) -> torch.Generator:
        """Create a seeded torch Generator on the configured device."""
        gen = torch.Generator(device=self.get_torch_device())
        gen.manual_seed(self.seed)
        return gen
