"""ConfigHelpers — static derived-computation helpers for chain count,
box size, and phi string formatting.

Reason these are code, not TOML: each is a small formula (compute_n_chains
returns a fixed count; compute_box_size applies the volume-fraction formula
with a dilute-regime floor; format_phi picks decimal digits based on phi
value).
"""

from rouse_model_python.src.libs.config.physical_constants import PhysicalConstants


class ConfigHelpers:
    @staticmethod
    def compute_n_chains(N: int, phi: float) -> int:
        """Fixed chain count for all state points.

        100 chains matches the Kuriata T1-T7 protocol (doubles Kuriata's
        original 50 to tighten statistical floor for +/-5% tolerances).
        """
        return 100

    @staticmethod
    def compute_box_size(N: int, n_chains: int, phi: float,
                         physics: PhysicalConstants) -> float:
        """Cubic box side length from volume fraction.

        L_box = (n_chains * N * sigma^3 / phi)^(1/3)

        A floor of 3 * sigma * N^0.588 (3x RMS end-to-end distance) applies
        only for dilute systems (phi < 0.01) where PBC self-interaction
        matters. In dense systems the box is full of chains and the floor
        is irrelevant.
        """
        sigma3 = physics.SIGMA ** 3
        box_from_phi = (n_chains * N * sigma3 / phi) ** (1.0 / 3.0)
        if phi < 0.01:
            rms_R = physics.SIGMA * (N ** 0.588)
            return max(box_from_phi, 3.0 * rms_R)
        return box_from_phi

    @staticmethod
    def format_phi(phi: float) -> str:
        """Format phi for directory/file names: 0.001, 0.01, 0.05, 0.10, 0.20, 0.30."""
        if phi < 0.01:
            return f"{phi:.3f}"
        return f"{phi:.2f}"
