"""ChainState — manages the full system state: positions tensor [n_chains, N, 3]."""

import math
import torch

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.number_space.number_space import NumberSpace
from rouse_model_python.src.libs.chain.segment_info import SegmentInfo


class ChainState:
    BOND_TOL = 0.05

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.device = cfg.get_torch_device()
        self.dtype = cfg.dtype
        self.ns = NumberSpace.from_config(cfg)

        self.positions = torch.zeros(
            cfg.n_chains, cfg.N, 3,
            dtype=self.dtype, device=self.device,
        )
        self.segments = SegmentInfo(cfg)
        self._l0 = cfg.l0
        self._bond_tol_sq = (cfg.l0 * (1.0 + self.BOND_TOL)) ** 2

    def initialize_serpentine(self, gen: torch.Generator):
        cfg = self.cfg
        ns = self.ns
        n_chains = cfg.n_chains
        N = cfg.N
        spacing = cfg.l0
        box = cfg.box_size

        chains_per_axis = math.ceil(n_chains ** (1.0 / 3.0))
        cell_size = box / chains_per_axis
        margin = spacing

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
                next_z = cz + z_dir * spacing
                if (z_dir > 0 and next_z <= z_max) or (z_dir < 0 and next_z >= z0):
                    cz = next_z
                else:
                    next_x = cx + x_dir * spacing
                    if (x_dir > 0 and next_x <= x_max) or (x_dir < 0 and next_x >= x0):
                        cx = next_x
                        z_dir = -z_dir
                    else:
                        cy += spacing
                        x_dir = -x_dir
                        z_dir = -z_dir

    def initialize_random_walk(self, gen: torch.Generator):
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
                    while True:
                        u = 2.0 * torch.rand(1, generator=gen, device=self.device, dtype=self.dtype).item() - 1.0
                        v = 2.0 * torch.rand(1, generator=gen, device=self.device, dtype=self.dtype).item() - 1.0
                        s = u * u + v * v
                        if 1e-10 < s < 1.0:
                            break
                    factor = 2.0 * math.sqrt(1.0 - s)
                    dx = u * factor
                    dy = v * factor
                    dz = 1.0 - 2.0 * s
                    cx += dx * spacing
                    cy += dy * spacing
                    cz += dz * spacing

    def wrap_all(self):
        self.positions = self.ns.wrap(self.positions)

    def get_chain_positions(self, chain_idx: int) -> torch.Tensor:
        return self.positions[chain_idx]

    def get_segment_positions(self, chain_idx: int, local_seg: int) -> torch.Tensor:
        s_start, s_end = self.segments.get_segment_range(chain_idx, local_seg)
        return self.positions[chain_idx, s_start:s_end, :]

    def apply_move(self, chain_idx: int, bead_start: int,
                   new_positions: torch.Tensor, validate: bool = True):
        n_beads = new_positions.shape[0]
        self.positions[chain_idx, bead_start:bead_start + n_beads, :] = new_positions
        if validate:
            self._validate_boundary_bonds(chain_idx, bead_start, n_beads)

    def apply_moves_batched(self,
                            chain_idx: torch.Tensor,
                            bead_start: torch.Tensor,
                            new_pos: torch.Tensor,
                            accepted: torch.Tensor,
                            n_moved: torch.Tensor) -> None:
        """Scatter accepted batched moves onto self.positions in a single call.

        Replaces the per-proposal `for i in range(B): state.apply_move(...)`
        loop. Safe when proposals within a batch act on disjoint beads (the
        segment-bucket permutation in perform_sweep guarantees this — each
        batch touches one segment per chain, and chains are disjoint).

        Args:
          chain_idx:   int64[B]          chain index per proposal (CUDA).
          bead_start:  int64[B]          start bead per proposal (CUDA).
          new_pos:     float[B,M,3]      new positions, padded to max_moved (CUDA).
          accepted:    bool[B]           accepted-mask output from the Metropolis
                                          step (CUDA).
          n_moved:     int64[B]          actual bead count per proposal; 0 = skip.

        All tensors must live on the same device as self.positions. No host
        syncs are introduced.
        """
        B, M = new_pos.shape[0], new_pos.shape[1]
        if B == 0 or M == 0:
            return
        device = self.device
        k_range = torch.arange(M, device=device, dtype=torch.int64)
        valid = (k_range.unsqueeze(0) < n_moved.unsqueeze(1)) & accepted.unsqueeze(1)
        chain_per_bead = chain_idx.unsqueeze(1).expand(B, M)
        bead_per_bead = bead_start.unsqueeze(1) + k_range.unsqueeze(0)
        flat_chain = chain_per_bead[valid]
        flat_bead = bead_per_bead[valid]
        flat_new_pos = new_pos[valid]
        if flat_chain.numel() == 0:
            return
        self.positions.index_put_((flat_chain, flat_bead), flat_new_pos)

    def _validate_boundary_bonds(self, chain_idx: int, bead_start: int, n_beads: int):
        ns = self.ns
        N = self.cfg.N
        tol_sq = self._bond_tol_sq
        pos = self.positions[chain_idx]
        if bead_start > 0:
            d2 = ns.mic_dist_sq(
                pos[bead_start - 1].unsqueeze(0),
                pos[bead_start].unsqueeze(0)).item()
            if d2 > tol_sq:
                raise RuntimeError(
                    f"Bond broken: chain {chain_idx}, beads "
                    f"{bead_start - 1}-{bead_start}, dist={d2**0.5:.4f} A")
        bead_end = bead_start + n_beads
        if bead_end < N:
            d2 = ns.mic_dist_sq(
                pos[bead_end - 1].unsqueeze(0),
                pos[bead_end].unsqueeze(0)).item()
            if d2 > tol_sq:
                raise RuntimeError(
                    f"Bond broken: chain {chain_idx}, beads "
                    f"{bead_end - 1}-{bead_end}, dist={d2**0.5:.4f} A")

    def get_all_flat(self) -> torch.Tensor:
        return self.positions.reshape(-1, 3)
