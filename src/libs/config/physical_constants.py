"""PhysicalConstants — dataclass loaded from configs/physics.toml."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class PhysicalConstants:
    SIGMA: float
    L0: float
    KB: float
    TEMPERATURE: float
    REPULSIVE_ENERGY: float
    CONTACT_ENERGY: float
    R_REP: float
    R_MIN: float
    R_MAX: float

    @property
    def KBT(self) -> float:
        return self.KB * self.TEMPERATURE

    @property
    def R_MAX_SQ(self) -> float:
        return self.R_MAX ** 2

    @property
    def R_REP_SQ(self) -> float:
        return self.R_REP ** 2

    @classmethod
    def from_toml(cls, path: Path) -> "PhysicalConstants":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        phys = data["physics"]
        cuts = data["cutoffs"]
        return cls(
            SIGMA=phys["sigma"],
            L0=phys["l0"],
            KB=phys["boltzmann_constant"],
            TEMPERATURE=phys["temperature"],
            REPULSIVE_ENERGY=phys["repulsive_energy"],
            CONTACT_ENERGY=phys["contact_energy"],
            R_REP=cuts["r_rep"],
            R_MIN=cuts["r_min"],
            R_MAX=cuts["r_max"],
        )
