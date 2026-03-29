"""
NumberSpace: central abstraction for periodic coordinate system operations.

Mirrors the C# NumberSpace class which manages coordinate conversions,
minimum image convention (MIC), and periodic boundary wrapping.

The C# version uses integer coordinates [0, 2^32) mapped to physical
Angstroms for SIMD stability. This PyTorch version works directly in
float64 Angstrom space, which provides equivalent precision (15-16
significant digits) without the integer mapping overhead.

All coordinate operations in the simulation flow through this class:
  - wrap():       enforce PBC, map positions to [-box/2, +box/2)
  - mic_delta():  minimum image displacement vector (r2 - r1, wrapped)
  - mic_dist_sq(): squared MIC distance
  - coord_to_cell(): map position to cell-list grid index
"""

import math
import torch
from typing import Optional


class NumberSpace:
    """
    Periodic cubic box coordinate system.

    Physical coordinate range: [-box_size/2, +box_size/2)
    All operations are vectorized PyTorch tensor ops for GPU compatibility.

    Attributes:
        box_size:   cubic box side length (Angstrom)
        half_box:   box_size / 2
        sigma:      bead diameter (Angstrom), used for reduced-unit conversions
        device:     torch device (cpu or cuda)
        dtype:      torch floating-point type (default float64)
    """

    def __init__(self, box_size: float, sigma: float = 3.8,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float64):
        self.box_size = box_size
        self.half_box = box_size / 2.0
        self.sigma = sigma
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Precompute inverse for fast division
        self._inv_box = 1.0 / box_size
        self._inv_sigma = 1.0 / sigma

    # ------------------------------------------------------------------
    # Core PBC operations (vectorized, GPU-ready)
    # ------------------------------------------------------------------

    def wrap(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Wrap positions into [-box/2, +box/2) under periodic boundary conditions.

        Equivalent to C# NumberSpace.WrapVec3Double().

        Args:
            positions: tensor of any shape with last dim = 3 (or scalar coords)

        Returns:
            wrapped positions, same shape
        """
        return positions - self.box_size * torch.floor(
            (positions + self.half_box) * self._inv_box
        )

    def wrap_scalar(self, v: float) -> float:
        """Wrap a single scalar coordinate (Python float). Used during init loops."""
        return v - self.box_size * math.floor((v + self.half_box) * self._inv_box)

    def mic_delta(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        """
        Minimum image convention displacement vector: r2 - r1, wrapped to
        [-box/2, +box/2).

        Equivalent to C# NumberSpace.GetMICDistanceDouble() applied per component.

        Args:
            r1, r2: tensors of matching shape (last dim = 3, or broadcastable)

        Returns:
            MIC displacement tensor, same shape as inputs
        """
        delta = r2 - r1
        return delta - self.box_size * torch.round(delta * self._inv_box)

    def mic_delta_scalar(self, v1: float, v2: float) -> float:
        """MIC displacement for a single component (Python float)."""
        d = v2 - v1
        return d - self.box_size * round(d * self._inv_box)

    def mic_dist_sq(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        """
        Squared MIC distance between position tensors.

        Equivalent to C# NumberSpace.GetSqrMICDistanceDouble() but for all
        three components at once.

        Args:
            r1, r2: [..., 3] position tensors

        Returns:
            [...] tensor of squared distances
        """
        d = self.mic_delta(r1, r2)
        return (d * d).sum(dim=-1)

    def mic_dist(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        """Euclidean MIC distance (not squared)."""
        return torch.sqrt(self.mic_dist_sq(r1, r2))

    # ------------------------------------------------------------------
    # Reduced-unit conversions (mirror C# ToReduced / FromReduced)
    # ------------------------------------------------------------------

    def to_reduced(self, angstrom: torch.Tensor) -> torch.Tensor:
        """Convert physical Angstrom coordinates to reduced (sigma) units."""
        return angstrom * self._inv_sigma

    def from_reduced(self, reduced: torch.Tensor) -> torch.Tensor:
        """Convert reduced (sigma) units to physical Angstrom coordinates."""
        return reduced * self.sigma

    def to_reduced_r2(self, r2_angstrom: float) -> float:
        """Convert squared distance from Angstrom² to reduced units."""
        return r2_angstrom * self._inv_sigma * self._inv_sigma

    # ------------------------------------------------------------------
    # Cell-list grid operations
    # ------------------------------------------------------------------

    def coord_to_cell(self, v: torch.Tensor, n_cells: int,
                       cell_size: float) -> torch.Tensor:
        """
        Map coordinates from [-box/2, +box/2) to cell indices [0, n_cells).

        Equivalent to C# CellList.CoordToCell().

        Args:
            v: coordinate tensor (any shape)
            n_cells: number of cells per axis
            cell_size: width of each cell

        Returns:
            integer tensor of cell indices, clamped to [0, n_cells-1]
        """
        shifted = v + self.half_box   # [0, box_size)
        c = (shifted / cell_size).long()
        return c.clamp(0, n_cells - 1)

    def coord_to_cell_scalar(self, v: float, n_cells: int,
                              cell_size: float) -> int:
        """Map a single scalar coordinate to a cell index."""
        shifted = v + self.half_box
        c = int(shifted / cell_size)
        return max(0, min(n_cells - 1, c))

    # ------------------------------------------------------------------
    # Chain unwrapping (for pivot moves and Rg² computation)
    # ------------------------------------------------------------------

    def unwrap_chain(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Unwrap a chain so that consecutive beads are within MIC distance.

        Takes positions of a single chain [N, 3] and returns unwrapped
        coordinates where bead i+1 is placed at the MIC-nearest image
        relative to bead i. The first bead stays at its original position.

        This is essential for:
          - Correct Rg² computation for chains that span the box boundary
          - Pivot move unwrapping before rotation

        Args:
            positions: [N, 3] wrapped positions of one chain

        Returns:
            [N, 3] unwrapped positions (may extend outside [-box/2, +box/2))
        """
        N = positions.shape[0]
        unwrapped = torch.empty_like(positions)
        unwrapped[0] = positions[0]

        # Sequential MIC chaining: each bead placed relative to previous
        for i in range(1, N):
            delta = positions[i] - positions[i - 1]
            delta = delta - self.box_size * torch.round(delta * self._inv_box)
            unwrapped[i] = unwrapped[i - 1] + delta

        return unwrapped

    def unwrap_chain_from_anchor(self, positions: torch.Tensor,
                                   anchor: torch.Tensor,
                                   forward: bool = True) -> torch.Tensor:
        """
        Unwrap a chain segment starting from an anchor point.

        Used by pivot moves: sequentially unwrap beads relative to the
        anchor (pivot bead), maintaining MIC between consecutive beads.

        Uses Python scalars for the sequential loop to avoid per-bead
        PyTorch dispatch overhead (~15μs/op × ~5 ops/bead = 75μs/bead).

        Args:
            positions: [K, 3] wrapped positions to unwrap (in chain order)
            anchor: [3] anchor position (e.g., pivot bead)
            forward: if True, unwrap positions[0] from anchor, then 1 from 0, etc.
                     if False, same but positions are already in the order to process

        Returns:
            [K, 3] unwrapped positions
        """
        K = positions.shape[0]
        box = self.box_size
        inv_box = self._inv_box

        # Convert to Python lists for zero-overhead iteration
        pos = positions.tolist()
        ax, ay, az = anchor.tolist() if anchor.dim() == 1 else [anchor[i].item() for i in range(3)]

        result = [[0.0, 0.0, 0.0]] * K

        # First bead: MIC from anchor
        dx = pos[0][0] - ax
        dy = pos[0][1] - ay
        dz = pos[0][2] - az
        dx -= box * round(dx * inv_box)
        dy -= box * round(dy * inv_box)
        dz -= box * round(dz * inv_box)
        px, py, pz = ax + dx, ay + dy, az + dz
        result[0] = [px, py, pz]

        # Subsequent beads: MIC from previous unwrapped position
        for k in range(1, K):
            dx = pos[k][0] - pos[k - 1][0]
            dy = pos[k][1] - pos[k - 1][1]
            dz = pos[k][2] - pos[k - 1][2]
            dx -= box * round(dx * inv_box)
            dy -= box * round(dy * inv_box)
            dz -= box * round(dz * inv_box)
            px, py, pz = px + dx, py + dy, pz + dz
            result[k] = [px, py, pz]

        return torch.tensor(result, dtype=positions.dtype, device=positions.device)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "NumberSpace":
        """Create NumberSpace from a SimulationConfig."""
        return cls(
            box_size=cfg.box_size,
            sigma=cfg.sigma,
            device=cfg.get_torch_device(),
            dtype=cfg.dtype,
        )
