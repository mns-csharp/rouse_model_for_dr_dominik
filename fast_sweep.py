"""
Fast MC sweep using numpy + numba for all hot-path operations.

Replaces multistep_mc.perform_sweep() for CPU sequential mode.
Key differences from the PyTorch path:
  1. Positions stored/manipulated as numpy arrays
  2. Numba-JIT'd energy computation (no PyTorch dispatch overhead)
  3. Scalar Rodrigues rotation (no tensor creation)
  4. All random numbers via numpy (no torch.rand sync)
  5. Intra-segment energy skipped (rigid-body invariance)
"""

import math
import numpy as np
from .fast_energy import FastEnergyComputer, _compute_segment_pair_energy_fast
from .chain import SegmentInfo


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOVE_SIZE = 20       # Batch size for multistep MC
REBUILD_INTERVAL = 3  # Rebuild cell list every N batches


# ---------------------------------------------------------------------------
# Scalar rotation helpers (zero allocation)
# ---------------------------------------------------------------------------

def _rodrigues_matrix(ux, uy, uz, angle):
    """Build 3x3 rotation matrix from axis + angle. Pure scalars → numpy."""
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return np.array([
        [t*ux*ux + c,      t*ux*uy - s*uz,  t*ux*uz + s*uy],
        [t*ux*uy + s*uz,   t*uy*uy + c,     t*uy*uz - s*ux],
        [t*ux*uz - s*uy,   t*uy*uz + s*ux,  t*uz*uz + c   ],
    ])


def _random_so3_matrix(rng):
    """Uniform random SO(3) rotation matrix using numpy RNG."""
    angle = 2.0 * math.pi * rng.random()
    while True:
        u = 2.0 * rng.random() - 1.0
        v = 2.0 * rng.random() - 1.0
        s2 = u * u + v * v
        if s2 < 1.0 and s2 > 1e-10:
            break
    factor = 2.0 * math.sqrt(1.0 - s2)
    ux = u * factor
    uy = v * factor
    uz = 1.0 - 2.0 * s2
    return _rodrigues_matrix(ux, uy, uz, angle)


def _mic_delta_3(ax, ay, az, bx, by, bz, box, inv_box):
    """Scalar MIC displacement."""
    dx = bx - ax; dy = by - ay; dz = bz - az
    dx -= box * round(dx * inv_box)
    dy -= box * round(dy * inv_box)
    dz -= box * round(dz * inv_box)
    return dx, dy, dz


def _wrap_pos(pos, box, half_box, inv_box):
    """Wrap positions into [-half_box, +half_box). In-place on numpy array."""
    return pos - box * np.floor((pos + half_box) * inv_box)


# Relative tolerance for bond-length validation (fraction of l0)
_BOND_TOL = 0.01


def _validate_boundary_bonds_np(positions_np, chain_idx, bead_start, n_moved,
                                 N, l0, box, inv_box):
    """Check bonds at the edges of a moved segment (numpy path).

    Raises RuntimeError if a boundary bond exceeds l0 * (1 + tol).
    """
    tol_sq = (l0 * (1.0 + _BOND_TOL)) ** 2
    pos = positions_np[chain_idx]

    if bead_start > 0:
        dx, dy, dz = _mic_delta_3(
            pos[bead_start - 1, 0], pos[bead_start - 1, 1], pos[bead_start - 1, 2],
            pos[bead_start, 0], pos[bead_start, 1], pos[bead_start, 2],
            box, inv_box)
        d2 = dx * dx + dy * dy + dz * dz
        if d2 > tol_sq:
            raise RuntimeError(
                f"Bond broken: chain {chain_idx}, beads "
                f"{bead_start - 1}-{bead_start}, "
                f"dist={d2**0.5:.4f} A (l0={l0:.1f} A)")

    bead_end = bead_start + n_moved
    if bead_end < N:
        dx, dy, dz = _mic_delta_3(
            pos[bead_end - 1, 0], pos[bead_end - 1, 1], pos[bead_end - 1, 2],
            pos[bead_end, 0], pos[bead_end, 1], pos[bead_end, 2],
            box, inv_box)
        d2 = dx * dx + dy * dy + dz * dz
        if d2 > tol_sq:
            raise RuntimeError(
                f"Bond broken: chain {chain_idx}, beads "
                f"{bead_end - 1}-{bead_end}, "
                f"dist={d2**0.5:.4f} A (l0={l0:.1f} A)")


# ---------------------------------------------------------------------------
# Move proposal (returns numpy arrays, no PyTorch)
# ---------------------------------------------------------------------------

AXIS_EPS = 1e-7


class FastProposal:
    """Lightweight move proposal using numpy arrays."""
    __slots__ = ['chain_idx', 'bead_start', 'n_moved',
                 'old_pos', 'new_pos', 'move_type']

    def __init__(self, chain_idx, bead_start, n_moved,
                 old_pos, new_pos, move_type):
        self.chain_idx = chain_idx
        self.bead_start = bead_start
        self.n_moved = n_moved
        self.old_pos = old_pos    # [M, 3] numpy
        self.new_pos = new_pos    # [M, 3] numpy
        self.move_type = move_type


def _unwrap_from_anchor(positions_np, anchor_idx, bead_start, bead_end,
                         box, inv_box):
    """
    Sequentially unwrap beads [bead_start, bead_end) outward from anchor_idx
    along the chain backbone.  Uses MIC between consecutive bonded beads
    (always correct since bonds << half_box), NOT single-hop MIC from the
    anchor (which fails when the segment spans more than half the box).

    Returns:
        unwrapped [M, 3] numpy array (may extend outside the box)
    """
    M = bead_end - bead_start
    unwrapped = np.empty((M, 3))

    if anchor_idx <= bead_start:
        # Anchor is at or before the segment — unwrap forward
        dx, dy, dz = _mic_delta_3(
            positions_np[anchor_idx, 0], positions_np[anchor_idx, 1],
            positions_np[anchor_idx, 2],
            positions_np[bead_start, 0], positions_np[bead_start, 1],
            positions_np[bead_start, 2], box, inv_box)
        unwrapped[0] = [positions_np[anchor_idx, 0] + dx,
                        positions_np[anchor_idx, 1] + dy,
                        positions_np[anchor_idx, 2] + dz]
        for k in range(1, M):
            gi = bead_start + k
            gi_prev = bead_start + k - 1
            dx = positions_np[gi, 0] - positions_np[gi_prev, 0]
            dy = positions_np[gi, 1] - positions_np[gi_prev, 1]
            dz = positions_np[gi, 2] - positions_np[gi_prev, 2]
            dx -= box * round(dx * inv_box)
            dy -= box * round(dy * inv_box)
            dz -= box * round(dz * inv_box)
            unwrapped[k] = [unwrapped[k-1, 0] + dx,
                            unwrapped[k-1, 1] + dy,
                            unwrapped[k-1, 2] + dz]
    else:
        # Anchor is after the segment — unwrap backward
        last = M - 1
        gi_last = bead_end - 1
        dx, dy, dz = _mic_delta_3(
            positions_np[anchor_idx, 0], positions_np[anchor_idx, 1],
            positions_np[anchor_idx, 2],
            positions_np[gi_last, 0], positions_np[gi_last, 1],
            positions_np[gi_last, 2], box, inv_box)
        unwrapped[last] = [positions_np[anchor_idx, 0] + dx,
                           positions_np[anchor_idx, 1] + dy,
                           positions_np[anchor_idx, 2] + dz]
        for k in range(last - 1, -1, -1):
            gi = bead_start + k
            gi_next = bead_start + k + 1
            dx = positions_np[gi, 0] - positions_np[gi_next, 0]
            dy = positions_np[gi, 1] - positions_np[gi_next, 1]
            dz = positions_np[gi, 2] - positions_np[gi_next, 2]
            dx -= box * round(dx * inv_box)
            dy -= box * round(dy * inv_box)
            dz -= box * round(dz * inv_box)
            unwrapped[k] = [unwrapped[k+1, 0] + dx,
                            unwrapped[k+1, 1] + dy,
                            unwrapped[k+1, 2] + dz]

    return unwrapped


def _rotate_and_wrap(unwrapped, anchor_pos, R, box, half_box, inv_box):
    """Rotate unwrapped positions around anchor, then wrap into box."""
    relative = unwrapped - anchor_pos[np.newaxis, :]
    rotated = relative @ R.T
    new_pos = anchor_pos[np.newaxis, :] + rotated
    return _wrap_pos(new_pos, box, half_box, inv_box)


def _propose_hinge(positions_np, chain_idx, seg_start, seg_end, N,
                   box, inv_box, half_box, max_angle, rng):
    """Propose hinge rotation. Uses sequential unwrapping from anchor."""
    a_idx = max(seg_start - 1, 0)
    b_idx = min(seg_end, N - 1)
    pos_a = positions_np[a_idx]

    old_pos = positions_np[seg_start:seg_end].copy()

    # Sequential unwrap from anchor along chain backbone
    unwrapped = _unwrap_from_anchor(positions_np, a_idx, seg_start, seg_end,
                                     box, inv_box)

    # Axis direction: from unwrapped anchor to unwrapped b_idx position.
    # b_idx is one bond past the segment end — unwrap it too via chain path.
    # Since we unwrapped forward from a_idx, extend one more bond to b_idx.
    dx = positions_np[b_idx, 0] - positions_np[seg_end - 1, 0]
    dy = positions_np[b_idx, 1] - positions_np[seg_end - 1, 1]
    dz = positions_np[b_idx, 2] - positions_np[seg_end - 1, 2]
    dx -= box * round(dx * inv_box)
    dy -= box * round(dy * inv_box)
    dz -= box * round(dz * inv_box)
    unwrapped_b = [unwrapped[-1, 0] + dx, unwrapped[-1, 1] + dy, unwrapped[-1, 2] + dz]

    # Axis = unwrapped_b - anchor (pos_a is already the anchor position)
    ax_dx = unwrapped_b[0] - pos_a[0]
    ax_dy = unwrapped_b[1] - pos_a[1]
    ax_dz = unwrapped_b[2] - pos_a[2]
    axis_len = math.sqrt(ax_dx*ax_dx + ax_dy*ax_dy + ax_dz*ax_dz)

    if axis_len < AXIS_EPS:
        return FastProposal(chain_idx, seg_start, seg_end - seg_start,
                            old_pos, old_pos.copy(), 'hinge')

    inv_len = 1.0 / axis_len
    ux, uy, uz = ax_dx * inv_len, ax_dy * inv_len, ax_dz * inv_len
    angle = (2.0 * rng.random() - 1.0) * max_angle
    R = _rodrigues_matrix(ux, uy, uz, angle)

    new_pos = _rotate_and_wrap(unwrapped, pos_a, R, box, half_box, inv_box)

    return FastProposal(chain_idx, seg_start, seg_end - seg_start,
                        old_pos, new_pos, 'hinge')


def _propose_tail(positions_np, chain_idx, seg_start, seg_end,
                  is_n_terminal, N, box, inv_box, half_box, max_angle, rng):
    """Propose tail rotation. Pure numpy/scalar."""
    if is_n_terminal:
        move_start = 0
        move_end = seg_end
        a_idx = min(seg_end, N - 1)
        b_idx = min(seg_end + 1, N - 1)
    else:
        move_start = seg_start
        move_end = N
        a_idx = max(seg_start - 1, 0)
        b_idx = max(seg_start - 2, 0)

    pos_a = positions_np[a_idx]
    pos_b = positions_np[b_idx]

    dx, dy, dz = _mic_delta_3(
        pos_a[0], pos_a[1], pos_a[2],
        pos_b[0], pos_b[1], pos_b[2], box, inv_box)
    axis_len = math.sqrt(dx*dx + dy*dy + dz*dz)

    if axis_len < AXIS_EPS:
        # Fallback: orthogonal axis
        c_idx = min(a_idx + 1, N - 1)
        tx, ty, tz = _mic_delta_3(
            positions_np[a_idx][0], positions_np[a_idx][1], positions_np[a_idx][2],
            positions_np[c_idx][0], positions_np[c_idx][1], positions_np[c_idx][2],
            box, inv_box)
        t_norm = math.sqrt(tx*tx + ty*ty + tz*tz)
        if t_norm > AXIS_EPS:
            if abs(tx) / t_norm > 0.9:
                rx, ry, rz = 0.0, 1.0, 0.0
            else:
                rx, ry, rz = 1.0, 0.0, 0.0
            dx = ty * rz - tz * ry
            dy = tz * rx - tx * rz
            dz = tx * ry - ty * rx
            axis_len = math.sqrt(dx*dx + dy*dy + dz*dz)

        if axis_len < AXIS_EPS:
            old_pos = positions_np[move_start:move_end].copy()
            mtype = 'n_tail' if is_n_terminal else 'c_tail'
            return FastProposal(chain_idx, move_start, move_end - move_start,
                                old_pos, old_pos.copy(), mtype)

    inv_len = 1.0 / axis_len
    ux, uy, uz = dx * inv_len, dy * inv_len, dz * inv_len
    angle = (2.0 * rng.random() - 1.0) * max_angle
    R = _rodrigues_matrix(ux, uy, uz, angle)

    old_pos = positions_np[move_start:move_end].copy()

    # Sequential unwrap from anchor, then rotate and wrap
    unwrapped = _unwrap_from_anchor(positions_np, a_idx, move_start, move_end,
                                     box, inv_box)
    new_pos = _rotate_and_wrap(unwrapped, pos_a, R, box, half_box, inv_box)

    mtype = 'n_tail' if is_n_terminal else 'c_tail'
    return FastProposal(chain_idx, move_start, move_end - move_start,
                        old_pos, new_pos, mtype)


def _propose_segment_move(positions_np, chain_idx, local_seg, seg_info,
                           N, box, inv_box, half_box, max_angle, rng):
    """Propose segment move (dispatch to hinge or tail)."""
    seg_type = seg_info.get_segment_type(local_seg)
    seg_start, seg_end = seg_info.get_segment_range(chain_idx, local_seg)

    if seg_type == SegmentInfo.INNER:
        return _propose_hinge(positions_np, chain_idx, seg_start, seg_end,
                              N, box, inv_box, half_box, max_angle, rng)
    elif seg_type == SegmentInfo.N_TERMINAL or seg_type == SegmentInfo.BOTH:
        return _propose_tail(positions_np, chain_idx, seg_start, seg_end,
                             True, N, box, inv_box, half_box, max_angle, rng)
    elif seg_type == SegmentInfo.C_TERMINAL:
        return _propose_tail(positions_np, chain_idx, seg_start, seg_end,
                             False, N, box, inv_box, half_box, max_angle, rng)
    else:
        raise ValueError(f"Unknown segment type: {seg_type}")


def _propose_pivot(positions_np, chain_idx, N, box, inv_box, half_box, rng):
    """Propose pivot move: SO(3) rotation of one side around random pivot."""
    if N <= 2:
        old_pos = positions_np[:N].copy()
        return FastProposal(chain_idx, 0, N, old_pos, old_pos.copy(), 'pivot')

    pivot_idx = 1 + int(rng.random() * (N - 2))
    pivot_idx = min(pivot_idx, N - 2)
    side = int(rng.random() * 2)

    if side == 0:
        rot_start, rot_end = 0, pivot_idx
    else:
        rot_start, rot_end = pivot_idx + 1, N

    n_rot = rot_end - rot_start
    if n_rot == 0:
        old_pos = positions_np[rot_start:rot_end].copy()
        return FastProposal(chain_idx, rot_start, 0,
                            old_pos, old_pos.copy(), 'pivot')

    anchor = positions_np[pivot_idx]

    # Get beads in processing order (reverse for N-terminal side)
    if side == 0:
        beads = positions_np[rot_start:rot_end][::-1]
    else:
        beads = positions_np[rot_start:rot_end]

    # Sequential PBC unwrapping from anchor (scalar loop)
    unwrapped = np.empty((n_rot, 3))
    # First bead from anchor
    dx, dy, dz = _mic_delta_3(
        anchor[0], anchor[1], anchor[2],
        beads[0, 0], beads[0, 1], beads[0, 2], box, inv_box)
    unwrapped[0] = [anchor[0] + dx, anchor[1] + dy, anchor[2] + dz]
    # Subsequent beads from previous
    for k in range(1, n_rot):
        dx = beads[k, 0] - beads[k-1, 0]
        dy = beads[k, 1] - beads[k-1, 1]
        dz = beads[k, 2] - beads[k-1, 2]
        dx -= box * round(dx * inv_box)
        dy -= box * round(dy * inv_box)
        dz -= box * round(dz * inv_box)
        unwrapped[k] = [unwrapped[k-1, 0] + dx,
                        unwrapped[k-1, 1] + dy,
                        unwrapped[k-1, 2] + dz]

    # Random SO(3) rotation
    R = _random_so3_matrix(rng)
    relative = unwrapped - anchor[np.newaxis, :]
    rotated = relative @ R.T
    new_unwrapped = anchor[np.newaxis, :] + rotated
    new_pos = _wrap_pos(new_unwrapped, box, half_box, inv_box)

    # Reverse back if N-terminal
    if side == 0:
        new_pos = new_pos[::-1].copy()

    old_pos = positions_np[rot_start:rot_end].copy()

    return FastProposal(chain_idx, rot_start, n_rot,
                        old_pos, new_pos, 'pivot')


# ---------------------------------------------------------------------------
# Metropolis acceptance
# ---------------------------------------------------------------------------

def _metropolis_accept(delta_e, kBT, rng):
    """Metropolis acceptance with overflow guards."""
    if delta_e <= 0.0:
        return True
    exponent = -delta_e / kBT
    if exponent <= -745.0:
        return False
    if exponent >= 709.0:
        return True
    return rng.random() < math.exp(exponent)


# ---------------------------------------------------------------------------
# Fast sweep
# ---------------------------------------------------------------------------

def fast_perform_sweep(positions_np, seg_info, energy_comp, cfg, stats, rng):
    """
    Perform one full MC sweep using numpy/numba fast path.

    Args:
        positions_np: [n_chains, N, 3] numpy array (modified in-place)
        seg_info: SegmentInfo
        energy_comp: FastEnergyComputer
        cfg: SimulationConfig
        stats: SimulationStats
        rng: numpy RandomState/Generator

    Phase 1: Batched multistep segment moves with rank-1 corrections
    Phase 2: Sequential pivot moves with cell-list delta-E
    """
    N = cfg.N
    n_chains = cfg.n_chains
    total_segs = seg_info.total_segments
    box = cfg.box_size
    inv_box = 1.0 / box
    half_box = box / 2.0
    kBT = cfg.kBT
    max_angle = cfg.max_angle_hinge

    # ── Build batched permutation with same-chain exclusion ──────────
    # Group segments into "rounds" so that each round contains at most
    # one segment per chain.  Within a round, segments from different
    # chains can be safely batched (moved in parallel from pre-batch
    # state) because accepting one move cannot break bonds of another
    # chain.  Rounds are processed sequentially, with positions fully
    # updated between rounds.
    #
    # Algorithm: bucket segments by chain, shuffle each bucket, then
    # interleave — round k gets the k-th segment from each chain.

    batch_size = MOVE_SIZE
    segs_per_chain = seg_info.segs_per_chain

    # Bucket segments by chain and shuffle within each bucket
    buckets = [[] for _ in range(n_chains)]
    for gs in range(total_segs):
        buckets[seg_info.seg_chain[gs]].append(gs)
    for bucket in buckets:
        rng.shuffle(bucket)

    # Build rounds: round k collects the k-th segment from each chain
    rounds = []
    for k in range(segs_per_chain):
        round_segs = [bucket[k] for bucket in buckets if k < len(bucket)]
        rng.shuffle(round_segs)  # randomize order within round
        rounds.append(round_segs)

    # Flatten rounds into perm; batch boundaries will respect rounds
    perm = []
    round_boundaries = []  # index in perm where each round starts
    for round_segs in rounds:
        round_boundaries.append(len(perm))
        perm.extend(round_segs)
    round_boundaries.append(len(perm))  # sentinel

    # Build batch ranges that never cross round boundaries
    batch_ranges = []
    for ri in range(len(rounds)):
        r_start = round_boundaries[ri]
        r_end = round_boundaries[ri + 1]
        for bs in range(r_start, r_end, batch_size):
            be = min(bs + batch_size, r_end)
            batch_ranges.append((bs, be))

    # ── Phase 1: Segment moves ──────────────────────────────────────
    for b_idx, (b_start, b_end) in enumerate(batch_ranges):

        # Rebuild cell list periodically
        if b_idx % REBUILD_INTERVAL == 0:
            pos_flat = positions_np.reshape(-1, 3)
            energy_comp.rebuild_cell_list(pos_flat)

        # Propose all moves in this batch
        proposals = []
        for idx in range(b_start, b_end):
            gs = perm[idx]
            chain_idx = seg_info.seg_chain[gs]
            local_seg = seg_info.seg_local[gs]
            chain_pos = positions_np[chain_idx]  # [N, 3] view
            p = _propose_segment_move(chain_pos, chain_idx, local_seg,
                                       seg_info, N, box, inv_box, half_box,
                                       max_angle, rng)
            proposals.append(p)

        B = len(proposals)

        # Compute delta-E for each proposal
        pos_flat = positions_np.reshape(-1, 3)
        delta_e = []
        for p in proposals:
            if p.n_moved == 0:
                delta_e.append(0.0)
            else:
                de = energy_comp.compute_delta_energy(
                    p.chain_idx, p.bead_start, p.n_moved,
                    p.old_pos, p.new_pos, N)
                delta_e.append(de)

        # Precompute cell-neighbor sets for proximity filtering
        cl = energy_comp.cell_list
        nc = cl._nc; inv_cs = cl._inv_cs
        hb = cl.half_box; nc_m1 = nc - 1
        neighbor_cells_table = cl._neighbor_cells

        cell_nbr_sets = []
        for p in proposals:
            if p.n_moved == 0:
                cell_nbr_sets.append(None)
                continue
            cells = set()
            for k in range(p.n_moved):
                cx = max(0, min(nc_m1, int((p.old_pos[k, 0] + hb) * inv_cs)))
                cy = max(0, min(nc_m1, int((p.old_pos[k, 1] + hb) * inv_cs)))
                cz = max(0, min(nc_m1, int((p.old_pos[k, 2] + hb) * inv_cs)))
                cells.add((cx * nc + cy) * nc + cz)
                cx = max(0, min(nc_m1, int((p.new_pos[k, 0] + hb) * inv_cs)))
                cy = max(0, min(nc_m1, int((p.new_pos[k, 1] + hb) * inv_cs)))
                cz = max(0, min(nc_m1, int((p.new_pos[k, 2] + hb) * inv_cs)))
                cells.add((cx * nc + cy) * nc + cz)
            expanded = set()
            for c in cells:
                expanded.update(neighbor_cells_table[c])
            cell_nbr_sets.append(expanded)

        # ── PRE-COMPUTE ALL ENERGY DATA (Migacz et al.) ─────────────
        # Per the paper: all energy information is computed BEFORE
        # any acceptance decision. The sequential acceptance loop
        # reads only pre-computed values.
        #
        # 1. delta_e[i]: already computed above (proposal i vs stationary system)
        # 2. E_mm[k,j]: 4 pairwise segment energy matrices for rank-1 corrections

        r_rep_sq = cfg.r_rep_sq
        rep_e = cfg.repulsive_energy

        # Pre-compute all 4 E_mm matrices: E00[i,j], E01[i,j], E10[i,j], E11[i,j]
        # E_mm[k,j] = pairwise energy between segments k and j
        # Using numba-JIT'd kernel for direct pairwise computation
        Emm00 = [[0.0]*B for _ in range(B)]
        Emm01 = [[0.0]*B for _ in range(B)]
        Emm10 = [[0.0]*B for _ in range(B)]
        Emm11 = [[0.0]*B for _ in range(B)]

        for i in range(B):
            if proposals[i].n_moved == 0:
                continue
            nbr_i = cell_nbr_sets[i]
            for j in range(i + 1, B):
                if proposals[j].n_moved == 0:
                    continue
                # Cell-proximity filter: skip pairs with disjoint neighborhoods
                nbr_j = cell_nbr_sets[j]
                if nbr_i is not None and nbr_j is not None and nbr_i.isdisjoint(nbr_j):
                    continue
                Emm00[i][j] = _compute_segment_pair_energy_fast(
                    proposals[i].old_pos, proposals[j].old_pos,
                    box, inv_box, r_rep_sq, rep_e)
                Emm01[i][j] = _compute_segment_pair_energy_fast(
                    proposals[i].old_pos, proposals[j].new_pos,
                    box, inv_box, r_rep_sq, rep_e)
                Emm10[i][j] = _compute_segment_pair_energy_fast(
                    proposals[i].new_pos, proposals[j].old_pos,
                    box, inv_box, r_rep_sq, rep_e)
                Emm11[i][j] = _compute_segment_pair_energy_fast(
                    proposals[i].new_pos, proposals[j].new_pos,
                    box, inv_box, r_rep_sq, rep_e)

        # ── SEQUENTIAL ACCEPTANCE (reads pre-computed values) ─────
        for i in range(B):
            if proposals[i].n_moved == 0:
                continue

            accepted = _metropolis_accept(delta_e[i], kBT, rng)
            stats.record(proposals[i].move_type, accepted)

            if not accepted:
                continue

            # Apply move
            p = proposals[i]
            positions_np[p.chain_idx, p.bead_start:p.bead_start + p.n_moved] = p.new_pos
            _validate_boundary_bonds_np(positions_np, p.chain_idx, p.bead_start,
                                         p.n_moved, N, cfg.l0, box, inv_box)

            # Rank-1 energy correction from pre-computed E_mm matrices
            for j in range(i + 1, B):
                if proposals[j].n_moved == 0:
                    continue
                e00 = Emm00[i][j]
                e01 = Emm01[i][j]
                e10 = Emm10[i][j]
                e11 = Emm11[i][j]
                correction = (e11 - e01) - (e10 - e00)
                if correction != 0.0:
                    delta_e[j] += correction

    # ── Phase 2: Pivot moves ────────────────────────────────────────
    pos_flat = positions_np.reshape(-1, 3)
    energy_comp.rebuild_cell_list(pos_flat)

    chain_perm = rng.permutation(n_chains)
    for c_idx in chain_perm:
        c = int(c_idx)
        chain_pos = positions_np[c]
        p = _propose_pivot(chain_pos, c, N, box, inv_box, half_box, rng)

        if p.n_moved == 0:
            continue

        de = energy_comp.compute_delta_energy(
            p.chain_idx, p.bead_start, p.n_moved,
            p.old_pos, p.new_pos, N)

        accepted = _metropolis_accept(de, kBT, rng)
        stats.record('pivot', accepted)

        if accepted:
            positions_np[p.chain_idx, p.bead_start:p.bead_start + p.n_moved] = p.new_pos
            _validate_boundary_bonds_np(positions_np, p.chain_idx, p.bead_start,
                                         p.n_moved, N, cfg.l0, box, inv_box)
