"""NumberSpace — periodic coordinate system: PBC wrapping, MIC distance,
cell-list grid mapping, chain unwrapping.

Pure PyTorch. No imports from other libs/.
"""

import math
from typing import Optional
import torch


class NumberSpace:
    def __init__(self, box_size: float, sigma: float = 3.8,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float64):
        self.box_size = box_size
        self.half_box = box_size / 2.0
        self.sigma = sigma
        if device is None:
            raise ValueError(
                "NumberSpace requires an explicit device. "
                "Use NumberSpace.from_config(cfg) or pass device=torch.device('cpu')."
            )
        self.device = device
        self.dtype = dtype
        self._inv_box = 1.0 / box_size
        self._inv_sigma = 1.0 / sigma

    # ── Core PBC ops ──────────────────────────────────────────────────
    def wrap(self, positions: torch.Tensor) -> torch.Tensor:
        return positions - self.box_size * torch.floor(
            (positions + self.half_box) * self._inv_box)

    def wrap_scalar(self, v: float) -> float:
        return v - self.box_size * math.floor((v + self.half_box) * self._inv_box)

    def mic_delta(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        delta = r2 - r1
        return delta - self.box_size * torch.round(delta * self._inv_box)

    def mic_delta_scalar(self, v1: float, v2: float) -> float:
        d = v2 - v1
        return d - self.box_size * round(d * self._inv_box)

    def mic_dist_sq(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        d = self.mic_delta(r1, r2)
        return (d * d).sum(dim=-1)

    def mic_dist(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mic_dist_sq(r1, r2))

    # ── Reduced-unit conversions ──────────────────────────────────────
    def to_reduced(self, angstrom: torch.Tensor) -> torch.Tensor:
        return angstrom * self._inv_sigma

    def from_reduced(self, reduced: torch.Tensor) -> torch.Tensor:
        return reduced * self.sigma

    def to_reduced_r2(self, r2_angstrom: float) -> float:
        return r2_angstrom * self._inv_sigma * self._inv_sigma

    # ── Cell-list grid ops ────────────────────────────────────────────
    def coord_to_cell(self, v: torch.Tensor, n_cells: int,
                      cell_size: float) -> torch.Tensor:
        shifted = v + self.half_box
        c = (shifted / cell_size).long()
        return c.clamp(0, n_cells - 1)

    def coord_to_cell_scalar(self, v: float, n_cells: int,
                             cell_size: float) -> int:
        shifted = v + self.half_box
        c = int(shifted / cell_size)
        return max(0, min(n_cells - 1, c))

    # ── Chain unwrapping ──────────────────────────────────────────────
    def unwrap_chain(self, positions: torch.Tensor) -> torch.Tensor:
        N = positions.shape[0]
        unwrapped = torch.empty_like(positions)
        unwrapped[0] = positions[0]
        for i in range(1, N):
            delta = positions[i] - positions[i - 1]
            delta = delta - self.box_size * torch.round(delta * self._inv_box)
            unwrapped[i] = unwrapped[i - 1] + delta
        return unwrapped

    def unwrap_chains(self, positions: torch.Tensor) -> torch.Tensor:
        bonds = self.mic_delta(positions[:, :-1, :], positions[:, 1:, :])
        cum_disp = torch.cumsum(bonds, dim=1)
        r0 = positions[:, 0:1, :]
        return torch.cat([r0, r0 + cum_disp], dim=1)

    def unwrap_chain_from_anchor(self, positions: torch.Tensor,
                                 anchor: torch.Tensor,
                                 forward: bool = True) -> torch.Tensor:
        K = positions.shape[0]
        box = self.box_size
        inv_box = self._inv_box
        d0 = positions[0:1] - anchor.unsqueeze(0)
        d0 = d0 - box * torch.round(d0 * inv_box)
        if K == 1:
            return anchor.unsqueeze(0) + d0
        bonds = positions[1:] - positions[:-1]
        bonds = bonds - box * torch.round(bonds * inv_box)
        all_deltas = torch.cat([d0, bonds], dim=0)
        return anchor.unsqueeze(0) + torch.cumsum(all_deltas, dim=0)

    def unwrap_segments_from_anchor(self, segments: torch.Tensor,
                                    anchors: torch.Tensor,
                                    valid: torch.Tensor) -> torch.Tensor:
        """Batched cumulative-MIC unwrap relative to a per-batch anchor.

        Equivalent to per-row unwrap_chain_from_anchor over a batch dim, with
        a valid-lane mask zeroing deltas in padding lanes so cumsum doesn't
        drift on them.

        Args:
          segments: [B, M, 3]  position lanes (padded to max_moved).
          anchors:  [B, 3]      anchor per row.
          valid:    [B, M] bool padding mask (True = real, False = pad).

        Returns:
          unwrapped: [B, M, 3]  with row's first valid bead = anchor + MIC(d0).
        """
        box = self.box_size
        inv_box = self._inv_box
        d0 = segments[:, 0:1, :] - anchors.unsqueeze(1)
        d0 = d0 - box * torch.round(d0 * inv_box)
        if segments.shape[1] == 1:
            return anchors.unsqueeze(1) + d0 * valid.unsqueeze(-1)
        bonds = segments[:, 1:, :] - segments[:, :-1, :]
        bonds = bonds - box * torch.round(bonds * inv_box)
        all_d = torch.cat([d0, bonds], dim=1) * valid.unsqueeze(-1)
        return anchors.unsqueeze(1) + torch.cumsum(all_d, dim=1)

    @classmethod
    def from_config(cls, cfg) -> "NumberSpace":
        return cls(
            box_size=cfg.box_size,
            sigma=cfg.sigma,
            device=cfg.get_torch_device(),
            dtype=cfg.dtype,
        )
