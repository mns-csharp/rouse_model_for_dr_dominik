"""Periodic bond-length rescaling.

Walks each chain from bead 0 outward and projects each successor onto
the l0-sphere of its predecessor. Preserves bond directions (hence all
planar and dihedral angles); only bond magnitudes snap back to l0.

Expected corrections are small (~0.05 A) when run every n_small_steps
sweeps. Called infrequently — loops are not hot.
"""

import math
import numpy as np
import torch


def rescale_bonds_numpy(positions_np: np.ndarray, l0: float,
                        box: float, half_box: float, inv_box: float) -> float:
    """In-place rescaling. positions_np shape: [n_chains, N, 3].

    Captures all original bond directions first, then walks each chain from
    bead 0 outward placing r[i+1] = r[i] + l0 * d_hat_orig[i].  Because the
    directions come from the PRE-rescale state, all bond directions are
    preserved exactly — planar and dihedral angles are unchanged.  Bead 0
    of each chain is pinned.

    Returns max |delta r| applied across all beads (for diagnostics).
    """
    n_chains, N, _ = positions_np.shape
    max_drift = 0.0
    for c in range(n_chains):
        pos = positions_np[c]
        bond_dirs = np.zeros((N - 1, 3))
        for i in range(N - 1):
            dx = pos[i + 1, 0] - pos[i, 0]
            dy = pos[i + 1, 1] - pos[i, 1]
            dz = pos[i + 1, 2] - pos[i, 2]
            dx -= box * round(dx * inv_box)
            dy -= box * round(dy * inv_box)
            dz -= box * round(dz * inv_box)
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            if length < 1e-12:
                continue
            inv_L = 1.0 / length
            bond_dirs[i, 0] = dx * inv_L
            bond_dirs[i, 1] = dy * inv_L
            bond_dirs[i, 2] = dz * inv_L
        for i in range(N - 1):
            new_x = pos[i, 0] + l0 * bond_dirs[i, 0]
            new_y = pos[i, 1] + l0 * bond_dirs[i, 1]
            new_z = pos[i, 2] + l0 * bond_dirs[i, 2]
            new_x -= box * math.floor((new_x + half_box) * inv_box)
            new_y -= box * math.floor((new_y + half_box) * inv_box)
            new_z -= box * math.floor((new_z + half_box) * inv_box)
            drx = new_x - pos[i + 1, 0]
            dry = new_y - pos[i + 1, 1]
            drz = new_z - pos[i + 1, 2]
            drx -= box * round(drx * inv_box)
            dry -= box * round(dry * inv_box)
            drz -= box * round(drz * inv_box)
            drift = math.sqrt(drx * drx + dry * dry + drz * drz)
            if drift > max_drift:
                max_drift = drift
            pos[i + 1, 0] = new_x
            pos[i + 1, 1] = new_y
            pos[i + 1, 2] = new_z
    return max_drift


def rescale_bonds_torch(positions: torch.Tensor, l0: float,
                        box: float, half_box: float) -> float:
    """In-place rescaling. positions shape: [n_chains, N, 3].

    Original bond directions are captured in one vectorised pass before
    any position is updated, so planar and dihedral angles are preserved.
    Vectorised across chains; sequential per bead. Returns max |delta r|.
    """
    inv_box = 1.0 / box
    n_chains, N, _ = positions.shape
    bond = positions[:, 1:, :] - positions[:, :-1, :]
    bond = bond - box * torch.round(bond * inv_box)
    lengths = torch.linalg.vector_norm(bond, dim=-1, keepdim=True).clamp(min=1e-12)
    bond_dirs = bond / lengths
    max_drift = torch.zeros((), dtype=positions.dtype, device=positions.device)
    for i in range(N - 1):
        new_pos = positions[:, i, :] + l0 * bond_dirs[:, i, :]
        new_pos = new_pos - box * torch.floor((new_pos + half_box) * inv_box)
        diff = new_pos - positions[:, i + 1, :]
        diff = diff - box * torch.round(diff * inv_box)
        drift = torch.linalg.vector_norm(diff, dim=-1).max()
        if drift > max_drift:
            max_drift = drift
        positions[:, i + 1, :] = new_pos
    return float(max_drift.item())
