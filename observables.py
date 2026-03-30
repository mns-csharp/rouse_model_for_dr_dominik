"""
Observable accumulators: R2, Rg2, middle-segment MSD, center-of-mass MSD,
and end-to-end vector autocorrelation.

All accumulation uses PyTorch tensors. Coordinate operations via NumberSpace.
"""

import torch
from .config import SimulationConfig
from .chain import ChainState
from .number_space import NumberSpace


class StaticObservables:
    """
    Computes per-chain R² (end-to-end distance squared) and Rg² (radius of
    gyration squared) from the current positions tensor.

    All MIC/unwrapping operations delegate to NumberSpace.
    """

    @staticmethod
    def compute_R2(positions: torch.Tensor, ns: NumberSpace) -> torch.Tensor:
        """
        End-to-end distance squared for each chain.

        Args:
            positions: [n_chains, N, 3]
            ns: NumberSpace for MIC computation

        Returns:
            [n_chains] tensor of R² values
        """
        unwrapped = ns.unwrap_chains(positions)
        diff = unwrapped[:, -1, :] - unwrapped[:, 0, :]
        return (diff * diff).sum(dim=-1)

    @staticmethod
    def compute_Rg2(positions: torch.Tensor, ns: NumberSpace) -> torch.Tensor:
        """
        Radius of gyration squared for each chain.

        Rg² = (1/N) Σ_i |r_i - r_cm|²

        Uses sequential bond-by-bond unwrapping to correctly handle chains
        that span (or wrap) the periodic box boundary.

        Returns:
            [n_chains] tensor of Rg² values
        """
        unwrapped = ns.unwrap_chains(positions)
        cm = unwrapped.mean(dim=1, keepdim=True)  # [n_chains, 1, 3]
        diff = unwrapped - cm
        return (diff * diff).sum(dim=2).mean(dim=1)

    @staticmethod
    def compute_end_to_end_vector(positions: torch.Tensor,
                                   ns: NumberSpace) -> torch.Tensor:
        """
        End-to-end vector R = r_end - r_start for each chain.

        Uses sequential unwrapping so the vector reflects the true chain
        path, not the minimum-image shortcut.

        Returns:
            [n_chains, 3] tensor
        """
        unwrapped = ns.unwrap_chains(positions)
        return unwrapped[:, -1, :] - unwrapped[:, 0, :]

    @staticmethod
    def compute_center_of_mass(positions: torch.Tensor,
                                ns: NumberSpace) -> torch.Tensor:
        """
        Center of mass for each chain, using sequential bond-by-bond
        unwrapping via NumberSpace.

        Returns:
            [n_chains, 3] tensor
        """
        unwrapped = ns.unwrap_chains(positions)
        return unwrapped.mean(dim=1)

    @staticmethod
    def compute_middle_segment_position(positions: torch.Tensor) -> torch.Tensor:
        """
        Position of the middle bead (index N//2) for each chain.

        Returns:
            [n_chains, 3] tensor
        """
        N = positions.shape[1]
        mid = N // 2
        return positions[:, mid, :].clone()


class DynamicAccumulator:
    """
    Accumulates time-lagged dynamic observables during production:
      - g1(t):  middle-segment MSD
      - gCM(t): center-of-mass MSD
      - gR(t):  end-to-end vector autocorrelation (normalized)

    MSD displacements computed via NumberSpace.mic_delta().
    """

    def __init__(self, cfg: SimulationConfig, ns: NumberSpace):
        self.cfg = cfg
        self.ns = ns
        self.device = cfg.get_torch_device()
        self.dtype = cfg.dtype
        self.n_chains = cfg.n_chains

        # Storage for reference snapshots
        self.mid_refs = []      # list of (sweep, [n_chains, 3])
        self.cm_refs = []       # list of (sweep, [n_chains, 3])
        self.ee_refs = []       # list of (sweep, [n_chains, 3])
        self.ee_norm_sq_sum = 0.0
        self.ee_norm_count = 0

        # Accumulators: dict[lag_sweep] -> [sum, count]
        self.g1_accum = {}
        self.gcm_accum = {}
        self.gr_accum = {}

    def record_snapshot(self, state: ChainState, sweep: int):
        """
        Record observables at a given production sweep.
        Called every sample_interval sweeps during production.
        """
        positions = state.positions
        ns = self.ns

        mid_pos = StaticObservables.compute_middle_segment_position(positions)
        cm_pos = StaticObservables.compute_center_of_mass(positions, ns)
        ee_vec = StaticObservables.compute_end_to_end_vector(positions, ns)

        # Accumulate |R|² for normalization
        ee_sq = (ee_vec * ee_vec).sum(dim=1)
        self.ee_norm_sq_sum += ee_sq.sum().item()
        self.ee_norm_count += self.n_chains

        # Compute vs all previous reference snapshots
        for ref_sweep, ref_mid in self.mid_refs:
            lag = sweep - ref_sweep
            d = ns.mic_delta(ref_mid, mid_pos)
            msd = (d * d).sum(dim=1).mean().item()
            if lag not in self.g1_accum:
                self.g1_accum[lag] = [0.0, 0]
            self.g1_accum[lag][0] += msd
            self.g1_accum[lag][1] += 1

        for ref_sweep, ref_cm in self.cm_refs:
            lag = sweep - ref_sweep
            d = ns.mic_delta(ref_cm, cm_pos)
            msd = (d * d).sum(dim=1).mean().item()
            if lag not in self.gcm_accum:
                self.gcm_accum[lag] = [0.0, 0]
            self.gcm_accum[lag][0] += msd
            self.gcm_accum[lag][1] += 1

        for ref_sweep, ref_ee in self.ee_refs:
            lag = sweep - ref_sweep
            dot = (ref_ee * ee_vec).sum(dim=1).mean().item()
            if lag not in self.gr_accum:
                self.gr_accum[lag] = [0.0, 0]
            self.gr_accum[lag][0] += dot
            self.gr_accum[lag][1] += 1

        # Store current as new reference
        self.mid_refs.append((sweep, mid_pos.clone()))
        self.cm_refs.append((sweep, cm_pos.clone()))
        self.ee_refs.append((sweep, ee_vec.clone()))

    def get_g1(self) -> dict:
        """Return {lag_sweep: mean_g1} for middle-segment MSD."""
        result = {}
        for lag, (s, c) in sorted(self.g1_accum.items()):
            if c > 0:
                result[lag] = s / c
        return result

    def get_gcm(self) -> dict:
        """Return {lag_sweep: mean_gCM} for center-of-mass MSD."""
        result = {}
        for lag, (s, c) in sorted(self.gcm_accum.items()):
            if c > 0:
                result[lag] = s / c
        return result

    def get_gr(self) -> dict:
        """
        Return {lag_sweep: normalized_gR} for end-to-end autocorrelation.
        Normalized by <|R(0)|²>.
        """
        if self.ee_norm_count == 0:
            return {}
        norm_sq = self.ee_norm_sq_sum / self.ee_norm_count
        if norm_sq < 1e-30:
            return {}

        result = {}
        for lag, (s, c) in sorted(self.gr_accum.items()):
            if c > 0:
                result[lag] = (s / c) / norm_sq
        return result


class SweepTracker:
    """
    Tracks per-sweep running averages of R² and Rg² during both
    equilibration and production (for static_vs_sweep.tsv).
    """

    def __init__(self, cfg: SimulationConfig, ns: NumberSpace):
        self.cfg = cfg
        self.ns = ns
        self.records = []  # list of (sweep, phase, mean_R2, mean_Rg2, ratio)

    def record(self, state: ChainState, sweep: int, phase: str):
        """Record running averages at this sweep."""
        positions = state.positions
        ns = self.ns

        R2 = StaticObservables.compute_R2(positions, ns)
        Rg2 = StaticObservables.compute_Rg2(positions, ns)

        mean_R2 = R2.mean().item()
        mean_Rg2 = Rg2.mean().item()
        ratio = mean_R2 / mean_Rg2 if mean_Rg2 > 1e-30 else 0.0

        self.records.append((sweep, phase, mean_R2, mean_Rg2, ratio))
