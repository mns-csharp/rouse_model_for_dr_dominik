"""
ChainState: manages positions tensor [n_chains, N, 3] with PBC utilities
and initialization (serpentine grid + random walk).

All coordinate operations delegate to NumberSpace.
"""

import math
import torch
from .config import SimulationConfig
from .number_space import NumberSpace


class SegmentInfo:
    """Precomputed segment boundaries and types for segmented multi-step MC."""

    # Segment types
    N_TERMINAL = 0
    C_TERMINAL = 1
    INNER = 2
    BOTH = 3  # single-segment chain

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        seg_size = cfg.residues_per_segment
        N = cfg.N
        n_chains = cfg.n_chains

        # Compute segment boundaries within a single chain
        segs_per_chain = cfg.n_segments_per_chain
        starts = []
        ends = []
        types = []
        for s in range(segs_per_chain):
            s_start = s * seg_size
            s_end = min((s + 1) * seg_size, N)
            starts.append(s_start)
            ends.append(s_end)
            if segs_per_chain == 1:
                types.append(self.BOTH)
            elif s == 0:
                types.append(self.N_TERMINAL)
            elif s == segs_per_chain - 1:
                types.append(self.C_TERMINAL)
            else:
                types.append(self.INNER)

        # Per-chain segment info (local bead indices)
        self.seg_starts = starts   # list of int, length = segs_per_chain
        self.seg_ends = ends       # list of int, length = segs_per_chain
        self.seg_types = types     # list of int, length = segs_per_chain
        self.segs_per_chain = segs_per_chain

        # Global segment indexing: segment_id = chain_idx * segs_per_chain + local_seg
        self.total_segments = n_chains * segs_per_chain

        # Precompute chain_idx and local_seg for each global segment
        self.seg_chain = []   # chain index for each global segment
        self.seg_local = []   # local segment index for each global segment
        for c in range(n_chains):
            for s in range(segs_per_chain):
                self.seg_chain.append(c)
                self.seg_local.append(s)

    def get_segment_range(self, chain_idx: int, local_seg: int):
        """Return (start, end) bead indices within the chain for a segment."""
        return self.seg_starts[local_seg], self.seg_ends[local_seg]

    def get_segment_type(self, local_seg: int) -> int:
        return self.seg_types[local_seg]

    def get_hinge_axis_indices(self, local_seg: int, N: int):
        """Return (axis_start_bead, axis_end_bead) for a hinge move.
        Axis is defined by the bead BEFORE and AFTER the segment."""
        s_start = self.seg_starts[local_seg]
        s_end = self.seg_ends[local_seg]
        # Axis start: bead before segment (or first bead if N-terminal)
        axis_start = max(s_start - 1, 0)
        # Axis end: bead after segment (or last bead if C-terminal)
        axis_end = min(s_end, N - 1)
        return axis_start, axis_end


class ChainState:
    """
    Manages the full system state: positions of all beads in all chains.

    Positions stored as tensor of shape [n_chains, N, 3] in physical
    Angstrom coordinates within [-box/2, +box/2).

    All PBC operations go through self.ns (NumberSpace).
    """

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.device = cfg.get_torch_device()
        self.dtype = cfg.dtype
        self.ns = NumberSpace.from_config(cfg)

        # Main positions tensor: [n_chains, N, 3]
        self.positions = torch.zeros(
            cfg.n_chains, cfg.N, 3,
            dtype=self.dtype, device=self.device
        )

        # Segment info
        self.segments = SegmentInfo(cfg)

    def initialize_serpentine(self, gen: torch.Generator):
        """Initialize chains on a 3D serpentine grid (matching C# LinearInit)."""
        cfg = self.cfg
        ns = self.ns
        n_chains = cfg.n_chains
        N = cfg.N
        spacing = cfg.l0  # 5.7 A
        box = cfg.box_size

        chains_per_axis = math.ceil(n_chains ** (1.0 / 3.0))
        cell_size = box / chains_per_axis
        margin = spacing  # keep beads away from cell edges

        for c in range(n_chains):
            gx = c % chains_per_axis
            gy = (c // chains_per_axis) % chains_per_axis
            gz = (c // (chains_per_axis * chains_per_axis)) % chains_per_axis

            x0 = gx * cell_size - ns.half_box + margin
            y0 = gy * cell_size - ns.half_box + margin
            z0 = gz * cell_size - ns.half_box + margin
            x_max = (gx + 1) * cell_size - ns.half_box - margin
            y_max = (gy + 1) * cell_size - ns.half_box - margin
            z_max = (gz + 1) * cell_size - ns.half_box - margin

            cx, cy, cz = x0, y0, z0
            z_dir, x_dir = 1, 1

            for i in range(N):
                self.positions[c, i, 0] = ns.wrap_scalar(cx)
                self.positions[c, i, 1] = ns.wrap_scalar(cy)
                self.positions[c, i, 2] = ns.wrap_scalar(cz)

                # Advance along Z
                next_z = cz + z_dir * spacing
                if (z_dir > 0 and next_z <= z_max) or (z_dir < 0 and next_z >= z0):
                    cz = next_z
                else:
                    # Try X, reverse Z
                    next_x = cx + x_dir * spacing
                    if (x_dir > 0 and next_x <= x_max) or (x_dir < 0 and next_x >= x0):
                        cx = next_x
                        z_dir = -z_dir
                    else:
                        # Step Y, reverse both
                        cy += spacing
                        x_dir = -x_dir
                        z_dir = -z_dir

    def initialize_random_walk(self, gen: torch.Generator):
        """Initialize chains as random walks (matching C# RandomWalkInit)."""
        cfg = self.cfg
        ns = self.ns
        n_chains = cfg.n_chains
        N = cfg.N
        spacing = cfg.l0
        box = cfg.box_size

        chains_per_axis = math.ceil(n_chains ** (1.0 / 3.0))
        cell_size = box / chains_per_axis

        for c in range(n_chains):
            gx = c % chains_per_axis
            gy = (c // chains_per_axis) % chains_per_axis
            gz = (c // (chains_per_axis * chains_per_axis)) % chains_per_axis

            cx = (gx + 0.5) * cell_size - ns.half_box
            cy = (gy + 0.5) * cell_size - ns.half_box
            cz = (gz + 0.5) * cell_size - ns.half_box

            for i in range(N):
                self.positions[c, i, 0] = ns.wrap_scalar(cx)
                self.positions[c, i, 1] = ns.wrap_scalar(cy)
                self.positions[c, i, 2] = ns.wrap_scalar(cz)

                if i < N - 1:
                    # Marsaglia method for uniform direction on sphere
                    while True:
                        u = 2.0 * torch.rand(1, generator=gen, device=self.device, dtype=self.dtype).item() - 1.0
                        v = 2.0 * torch.rand(1, generator=gen, device=self.device, dtype=self.dtype).item() - 1.0
                        s = u * u + v * v
                        if s < 1.0 and s > 1e-10:
                            break
                    factor = 2.0 * math.sqrt(1.0 - s)
                    dx = u * factor
                    dy = v * factor
                    dz = 1.0 - 2.0 * s

                    cx += dx * spacing
                    cy += dy * spacing
                    cz += dz * spacing

    def wrap_all(self):
        """Wrap all positions into the periodic box."""
        self.positions = self.ns.wrap(self.positions)

    def get_chain_positions(self, chain_idx: int) -> torch.Tensor:
        """Return positions for one chain: shape [N, 3]."""
        return self.positions[chain_idx]

    def get_segment_positions(self, chain_idx: int, local_seg: int) -> torch.Tensor:
        """Return positions for one segment: shape [seg_size, 3]."""
        s_start, s_end = self.segments.get_segment_range(chain_idx, local_seg)
        return self.positions[chain_idx, s_start:s_end, :]

    def apply_move(self, chain_idx: int, bead_start: int, new_positions: torch.Tensor):
        """Apply accepted move: overwrite positions for a contiguous bead range."""
        n_beads = new_positions.shape[0]
        self.positions[chain_idx, bead_start:bead_start + n_beads, :] = new_positions

    def get_all_flat(self) -> torch.Tensor:
        """Return all positions as flat tensor [total_beads, 3]."""
        return self.positions.reshape(-1, 3)
