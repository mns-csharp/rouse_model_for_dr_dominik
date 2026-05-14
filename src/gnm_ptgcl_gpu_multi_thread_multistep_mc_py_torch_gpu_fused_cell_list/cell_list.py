"""GPU-resident cell list for the spatial decomposition of bead positions.

Given a (NK, 3) tensor of bead positions wrapped into [-half_box, half_box),
divides the box into cubic cells of side `r_cell` and bins beads into cells.

Two outputs power the per-batch neighbour lookup:
  - `idx_sorted` : (NK,) int64. Bead indices sorted by their cell_id.
  - `cell_starts`: (n_cells + 1,) int64. cell_starts[c]..cell_starts[c+1] is
                   the slice of `idx_sorted` holding beads in cell `c`.

For each moved bead, the energy kernel:
  1. Computes the moved bead's cell.
  2. Walks the 27 neighbour cells.
  3. Concatenates their cell-slices into a (max_neighbors,) candidate-index row.

The 27-cell walk produces variable-length rows; we pad with -1 to a fixed
`max_neighbors` so the downstream cdist + zone-energy reduction stays
shape-static (friendly to torch.compile).

Periodic boundary conditions: cell indices wrap via modulo `n_cells_per_dim`.
"""

from __future__ import annotations

import torch


def _wrap_into_box(positions: torch.Tensor, box: float) -> torch.Tensor:
    """Wrap positions from [-half_box, half_box) (or any) into [0, box)."""
    return positions - box * torch.floor(positions / box)


def build_cell_index(positions: torch.Tensor, box: float, r_cell: float):
    """Sort beads by cell_id; return (idx_sorted, cell_starts, n_cells_per_dim).

    positions: (NK, 3) float, may be wrapped or unwrapped (we wrap inside).
    Returns:
      idx_sorted   : (NK,) long, original bead indices sorted by cell_id
      cell_starts  : (n_cells + 1,) long, with cell_starts[-1] == NK
      n_cells_per_dim : int
    """
    device = positions.device
    NK = positions.shape[0]
    n_cells_per_dim = max(1, int(box / r_cell))   # integer cells per axis
    n_cells = n_cells_per_dim ** 3

    pos_wrapped = _wrap_into_box(positions, box)
    cell_xyz = (pos_wrapped / (box / n_cells_per_dim)).long()
    cell_xyz.clamp_(0, n_cells_per_dim - 1)
    cell_id = (cell_xyz[:, 0]
               + cell_xyz[:, 1] * n_cells_per_dim
               + cell_xyz[:, 2] * n_cells_per_dim ** 2)

    # Sort beads by cell_id (stable). idx_sorted[i] = original index of i-th
    # sorted bead.
    sort_vals, idx_sorted = torch.sort(cell_id, stable=True)

    # cell_starts: prefix count of beads per cell. Use bincount + cumsum.
    counts = torch.bincount(sort_vals, minlength=n_cells)
    cell_starts = torch.zeros(n_cells + 1, dtype=torch.long, device=device)
    cell_starts[1:] = torch.cumsum(counts, dim=0)

    return idx_sorted, cell_starts, n_cells_per_dim


def _cell_of_position(positions: torch.Tensor, box: float,
                       n_cells_per_dim: int) -> torch.Tensor:
    """Return cell_xyz (P, 3) int for each of P positions."""
    pos_wrapped = _wrap_into_box(positions, box)
    cell_xyz = (pos_wrapped / (box / n_cells_per_dim)).long()
    cell_xyz.clamp_(0, n_cells_per_dim - 1)
    return cell_xyz


# Precomputed (27, 3) offset tensor — cached per-device so torch.compile sees
# a stable constant after the first call.
_NEIGHBOR_OFFSETS_CACHE: dict = {}


def _neighbor_offsets(device):
    key = str(device)
    if key not in _NEIGHBOR_OFFSETS_CACHE:
        ofs = torch.tensor(
            [[dx, dy, dz] for dx in (-1, 0, 1)
                          for dy in (-1, 0, 1)
                          for dz in (-1, 0, 1)],
            dtype=torch.long, device=device,
        )
        _NEIGHBOR_OFFSETS_CACHE[key] = ofs
    return _NEIGHBOR_OFFSETS_CACHE[key]


def gather_candidates(query_positions: torch.Tensor,
                      idx_sorted: torch.Tensor,
                      cell_starts: torch.Tensor,
                      n_cells_per_dim: int,
                      box: float,
                      max_neighbors: int) -> torch.Tensor:
    """For each of P query positions, return (P, max_neighbors) int tensor of
    candidate-neighbour bead indices, padded with -1.

    Walks the 27-cell neighbourhood (the query's cell + 26 surrounding cells,
    with PBC wrap). Per-cell bead counts beyond `max_neighbors` are silently
    truncated — pass a conservative ceiling.

    GPU-only operations; no host transfer. Works for any P.
    """
    device = query_positions.device
    P = query_positions.shape[0]
    if P == 0:
        return torch.empty((0, max_neighbors), dtype=torch.long, device=device)

    cell_xyz = _cell_of_position(query_positions, box, n_cells_per_dim)  # (P, 3)
    offsets = _neighbor_offsets(device)                                  # (27, 3)

    # (P, 27, 3) neighbour cell coordinates with PBC wrap.
    n_xyz = (cell_xyz.unsqueeze(1) + offsets.unsqueeze(0)) % n_cells_per_dim
    n_ids = (n_xyz[..., 0]
             + n_xyz[..., 1] * n_cells_per_dim
             + n_xyz[..., 2] * n_cells_per_dim ** 2)                     # (P, 27)

    # For each (p, c) cell-id, fetch [cell_starts[c], cell_starts[c+1]).
    # We use a fully-vectorised approach: for each of the (P, 27) cells, pull
    # the first up-to-K_per_cell beads where K_per_cell = max_neighbors // 27 + 1
    # — but this can miss beads in dense cells. Safer: gather per-cell counts,
    # build a flat (P, 27, max_per_cell) indices tensor.
    starts = cell_starts[n_ids]                                          # (P, 27)
    ends = cell_starts[n_ids + 1]                                        # (P, 27)
    counts = ends - starts                                               # (P, 27)

    # Per-position lane index 0..max_per_cell-1; expand to (P, 27, max_per_cell).
    # max_per_cell is the largest single-cell occupancy we expect; we cap at
    # ceil(max_neighbors / 27). For phi=0.10 typical occupancy is small.
    max_per_cell = max(1, (max_neighbors + 26) // 27)
    lane = torch.arange(max_per_cell, device=device)                     # (max_per_cell,)
    # within-cell flat offset; -1 padding when lane >= count.
    lane_b = lane.view(1, 1, -1).expand(P, 27, -1)                       # (P, 27, max_per_cell)
    valid = lane_b < counts.unsqueeze(-1)
    flat_pos = starts.unsqueeze(-1) + lane_b                             # (P, 27, max_per_cell)
    flat_pos_clamped = flat_pos.clamp(max=idx_sorted.shape[0] - 1)
    cand = idx_sorted[flat_pos_clamped]                                  # (P, 27, max_per_cell)
    cand = torch.where(valid, cand, torch.full_like(cand, -1))

    # Flatten 27 × max_per_cell → max_neighbors (truncate if necessary).
    cand_flat = cand.reshape(P, 27 * max_per_cell)
    if cand_flat.shape[1] >= max_neighbors:
        cand_flat = cand_flat[:, :max_neighbors]
    else:
        pad = torch.full((P, max_neighbors - cand_flat.shape[1]), -1,
                         dtype=torch.long, device=device)
        cand_flat = torch.cat([cand_flat, pad], dim=1)
    return cand_flat
