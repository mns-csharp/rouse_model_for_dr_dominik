"""
Observable accumulators: R2, Rg2, middle-segment MSD, center-of-mass MSD,
and end-to-end vector autocorrelation.

All accumulation uses PyTorch tensors. Coordinate operations via NumberSpace.

IMPORTANT: MSD computation tracks cumulative per-bead displacements.
Each bead's displacement between consecutive snapshots is obtained via
mic_delta on wrapped positions (always small relative to box/2).  These
per-bead increments are accumulated, then the CM displacement is the
average over all beads.  This avoids artefacts from chain-following
unwrapping (whose CM depends on internal conformation, not diffusion).
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
        unwrapped = ns.unwrap_chains(positions)
        diff = unwrapped[:, -1, :] - unwrapped[:, 0, :]
        return (diff * diff).sum(dim=-1)

    @staticmethod
    def compute_Rg2(positions: torch.Tensor, ns: NumberSpace) -> torch.Tensor:
        unwrapped = ns.unwrap_chains(positions)
        cm = unwrapped.mean(dim=1, keepdim=True)
        diff = unwrapped - cm
        return (diff * diff).sum(dim=2).mean(dim=1)

    @staticmethod
    def compute_end_to_end_vector(positions: torch.Tensor,
                                   ns: NumberSpace) -> torch.Tensor:
        unwrapped = ns.unwrap_chains(positions)
        return unwrapped[:, -1, :] - unwrapped[:, 0, :]

    @staticmethod
    def compute_center_of_mass(positions: torch.Tensor,
                                ns: NumberSpace) -> torch.Tensor:
        unwrapped = ns.unwrap_chains(positions)
        return unwrapped.mean(dim=1)

    @staticmethod
    def compute_middle_segment_position(positions: torch.Tensor,
                                         ns: NumberSpace) -> torch.Tensor:
        unwrapped = ns.unwrap_chains(positions)
        N = positions.shape[1]
        mid = N // 2
        return unwrapped[:, mid, :].clone()


class DynamicAccumulator:
    """
    Accumulates time-lagged dynamic observables during production:
      - g1(t):  middle-segment MSD
      - gCM(t): center-of-mass MSD
      - gR(t):  end-to-end vector autocorrelation (normalized)

    MSD uses per-bead cumulative displacement tracking:
      - At each snapshot, mic_delta of every bead's WRAPPED position from
        the previous snapshot gives the true per-bead displacement (small
        relative to box/2, so MIC is exact).
      - Middle-bead displacement is accumulated for g1.
      - Mean over all beads gives CM displacement for g_CM.
      - This avoids artefacts from chain-following unwrapping, where pivots
        change the apparent CM even though the physical CM doesn't move.
    """

    def __init__(self, cfg: SimulationConfig, ns: NumberSpace):
        self.cfg = cfg
        self.ns = ns
        self.device = cfg.get_torch_device()
        self.dtype = cfg.dtype
        self.n_chains = cfg.n_chains
        self.N = cfg.N

        # Previous snapshot: WRAPPED all-bead positions [n_chains, N, 3]
        self._prev_positions = None

        # Cumulative per-bead displacement [n_chains, N, 3]
        self._cum_bead_disp = None

        # Derived cumulative quantities (computed from _cum_bead_disp)
        # These are updated each snapshot for storage in reference lists.

        # Storage for reference snapshots
        self.mid_refs = []      # list of (sweep, cum_mid [n_chains, 3])
        self.cm_refs = []       # list of (sweep, cum_cm [n_chains, 3])
        self.ee_refs = []       # list of (sweep, ee_vec [n_chains, 3])
        self.ee_norm_sq_sum = 0.0
        self.ee_norm_count = 0

        # Accumulators: dict[lag_sweep] -> [sum, count]
        self.g1_accum = {}
        self.gcm_accum = {}
        self.gr_accum = {}

        self._snapshot_count = 0
        self._max_refs = 500
        self._ref_modulo = 1

    def record_snapshot(self, state: ChainState, sweep: int,
                        cum_bead_disp: "torch.Tensor | None" = None):
        """Record observables at a given production sweep.

        Args:
            state: current chain state (wrapped positions)
            sweep: production sweep number
            cum_bead_disp: [n_chains, N, 3] cumulative per-bead displacement
                from local (segment) moves only.  If None, falls back to
                computing displacement from wrapped positions (includes all
                move types).
        """
        positions = state.positions  # [n_chains, N, 3], WRAPPED
        ns = self.ns
        mid_idx = self.N // 2

        if cum_bead_disp is not None:
            # Use externally-tracked cumulative displacement (segment moves only)
            pass
        else:
            # Fallback: compute from positions (includes ALL move types)
            if self._prev_positions is None:
                self._cum_bead_disp = torch.zeros_like(positions)
                self._prev_positions = positions.clone()
            else:
                delta = ns.mic_delta(self._prev_positions, positions)
                self._cum_bead_disp = self._cum_bead_disp + delta
                self._prev_positions = positions.clone()
            cum_bead_disp = self._cum_bead_disp

        # Cumulative middle-bead displacement
        cum_mid = cum_bead_disp[:, mid_idx, :].clone()  # [n_chains, 3]
        # Cumulative CM displacement = mean over all beads
        cum_cm = cum_bead_disp.mean(dim=1).clone()  # [n_chains, 3]

        # End-to-end vector (correctly uses chain-following unwrap for internal geometry)
        ee_vec = StaticObservables.compute_end_to_end_vector(positions, ns)

        # Accumulate |R|² for normalization
        ee_sq = (ee_vec * ee_vec).sum(dim=1)
        self.ee_norm_sq_sum += ee_sq.sum().item()
        self.ee_norm_count += self.n_chains

        # Compute MSD vs all previous reference snapshots
        for ref_sweep, ref_mid in self.mid_refs:
            lag = sweep - ref_sweep
            d = cum_mid - ref_mid
            msd = (d * d).sum(dim=1).mean().item()
            if lag not in self.g1_accum:
                self.g1_accum[lag] = [0.0, 0]
            self.g1_accum[lag][0] += msd
            self.g1_accum[lag][1] += 1

        for ref_sweep, ref_cm in self.cm_refs:
            lag = sweep - ref_sweep
            d = cum_cm - ref_cm
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

        # Store current as new reference (sub-sample if too many snapshots)
        self._snapshot_count += 1
        total_expected = self.cfg.prod_sweeps // self.cfg.sample_interval
        if total_expected > self._max_refs:
            self._ref_modulo = max(1, total_expected // self._max_refs)
        if self._snapshot_count % self._ref_modulo == 0 or self._snapshot_count <= 10:
            self.mid_refs.append((sweep, cum_mid))
            self.cm_refs.append((sweep, cum_cm))
            self.ee_refs.append((sweep, ee_vec.clone()))

    def update_prev_positions(self, state: ChainState):
        """Update the previous-positions reference to the CURRENT wrapped
        positions.  Call this AFTER pivot moves so that the next snapshot's
        mic_delta only captures local-move displacement."""
        if self._prev_positions is not None:
            self._prev_positions = state.positions.clone()

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
        self.records = []

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
