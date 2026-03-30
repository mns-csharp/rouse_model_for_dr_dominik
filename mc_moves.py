"""
Monte Carlo moves: segmented hinge, N-tail, C-tail, and pivot.

All rotations use the Rodrigues formula. Positions are in physical Angstrom
coordinates with periodic boundary conditions via NumberSpace.
"""

import math
import torch
from .config import SimulationConfig
from .chain import ChainState, SegmentInfo
from .number_space import NumberSpace


AXIS_EPS = 1e-7  # Minimum axis length to avoid degenerate rotations


def rodrigues_rotation_matrix(axis: torch.Tensor, angle: float,
                               dtype: torch.dtype = torch.float64,
                               device: torch.device = None) -> torch.Tensor:
    """
    Build 3x3 rotation matrix via Rodrigues formula.

    R = cos(θ)·I + sin(θ)·K + (1 - cos(θ))·(u ⊗ u)

    Uses Python scalar math to build the matrix, avoiding ~15 tiny tensor
    operations (each with ~15μs dispatch overhead).
    """
    if device is None:
        device = axis.device

    # Extract axis components as scalars
    ux = axis[0].item()
    uy = axis[1].item()
    uz = axis[2].item()
    norm = math.sqrt(ux * ux + uy * uy + uz * uz)

    if norm < AXIS_EPS:
        return torch.eye(3, dtype=dtype, device=device)

    inv_norm = 1.0 / norm
    ux *= inv_norm
    uy *= inv_norm
    uz *= inv_norm

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c

    # Build matrix directly from scalar formula
    return torch.tensor([
        [t * ux * ux + c,      t * ux * uy - s * uz,  t * ux * uz + s * uy],
        [t * ux * uy + s * uz, t * uy * uy + c,       t * uy * uz - s * ux],
        [t * ux * uz - s * uy, t * uy * uz + s * ux,  t * uz * uz + c     ],
    ], dtype=dtype, device=device)


def random_so3_matrix(gen: torch.Generator, dtype: torch.dtype = torch.float64,
                       device: torch.device = None) -> torch.Tensor:
    """
    Generate a uniform random SO(3) rotation matrix.

    Uses the subgroup algorithm: random axis on S² + random angle [0, 2π).
    Random numbers from torch generator; matrix built with Python scalars.
    """
    angle = 2.0 * math.pi * torch.rand(1, generator=gen, dtype=dtype, device=device).item()

    # Random axis via Marsaglia method
    while True:
        u = 2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0
        v = 2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0
        s = u * u + v * v
        if s < 1.0 and s > 1e-10:
            break
    factor = 2.0 * math.sqrt(1.0 - s)
    ux = u * factor
    uy = v * factor
    uz = 1.0 - 2.0 * s

    # Build rotation matrix directly (scalar Rodrigues)
    c = math.cos(angle)
    si = math.sin(angle)
    t = 1.0 - c

    return torch.tensor([
        [t * ux * ux + c,       t * ux * uy - si * uz,  t * ux * uz + si * uy],
        [t * ux * uy + si * uz, t * uy * uy + c,        t * uy * uz - si * ux],
        [t * ux * uz - si * uy, t * uy * uz + si * ux,  t * uz * uz + c      ],
    ], dtype=dtype, device=device)


def apply_rotation_to_beads_unwrapped(unwrapped: torch.Tensor,
                                       center: torch.Tensor,
                                       R: torch.Tensor,
                                       ns: NumberSpace) -> torch.Tensor:
    """
    Rotate pre-unwrapped positions around a center point using rotation
    matrix R, then wrap back into the periodic box.

    Args:
        unwrapped: [n_beads, 3] sequentially unwrapped positions
        center: [3] rotation center (anchor bead position)
        R: [3, 3] rotation matrix
        ns: NumberSpace for PBC wrapping

    Returns:
        [n_beads, 3] new wrapped positions
    """
    relative = unwrapped - center.unsqueeze(0)
    rotated = torch.mm(relative, R.t())
    new_pos = center.unsqueeze(0) + rotated
    return ns.wrap(new_pos)


class MoveProposal:
    """Result of a proposed MC move."""
    __slots__ = ['chain_idx', 'bead_start', 'bead_end',
                 'old_positions', 'new_positions', 'move_type']

    def __init__(self, chain_idx: int, bead_start: int, bead_end: int,
                 old_positions: torch.Tensor, new_positions: torch.Tensor,
                 move_type: str):
        self.chain_idx = chain_idx
        self.bead_start = bead_start
        self.bead_end = bead_end
        self.old_positions = old_positions
        self.new_positions = new_positions
        self.move_type = move_type

    @property
    def n_moved(self) -> int:
        return self.bead_end - self.bead_start


def _mic_delta_scalar(p1, p2, box, inv_box):
    """MIC delta between two [3] tensors using Python scalars. Returns [dx,dy,dz] tuple."""
    ax, ay, az = p1[0].item(), p1[1].item(), p1[2].item()
    bx, by, bz = p2[0].item(), p2[1].item(), p2[2].item()
    dx = bx - ax; dy = by - ay; dz = bz - az
    dx -= box * round(dx * inv_box)
    dy -= box * round(dy * inv_box)
    dz -= box * round(dz * inv_box)
    return dx, dy, dz


def propose_hinge_move(state: ChainState, chain_idx: int, seg_start: int,
                        seg_end: int, gen: torch.Generator,
                        cfg: SimulationConfig) -> MoveProposal:
    """
    Propose a hinge rotation: rotate beads [seg_start, seg_end) around the
    axis defined by the bead BEFORE and AFTER the segment.
    """
    N = cfg.N
    ns = state.ns
    device = cfg.get_torch_device()
    dtype = cfg.dtype
    positions = state.positions[chain_idx]  # [N, 3]

    # Axis endpoints (clamp to chain boundaries)
    axis_bead_a = max(seg_start - 1, 0)
    axis_bead_b = min(seg_end, N - 1)

    pos_a = positions[axis_bead_a]  # [3]
    old_pos = positions[seg_start:seg_end].clone()

    # Sequential unwrap from anchor along chain backbone
    unwrapped = ns.unwrap_chain_from_anchor(old_pos, pos_a)

    # Axis direction: unwrap bead_b via one more bond past the segment end
    box = ns.box_size
    inv_box = ns._inv_box
    dx_b, dy_b, dz_b = _mic_delta_scalar(
        positions[seg_end - 1], positions[axis_bead_b], box, inv_box)
    uw_b_x = unwrapped[-1, 0].item() + dx_b
    uw_b_y = unwrapped[-1, 1].item() + dy_b
    uw_b_z = unwrapped[-1, 2].item() + dz_b
    dx = uw_b_x - pos_a[0].item()
    dy = uw_b_y - pos_a[1].item()
    dz = uw_b_z - pos_a[2].item()
    axis_len = math.sqrt(dx * dx + dy * dy + dz * dz)

    if axis_len < AXIS_EPS:
        return MoveProposal(chain_idx, seg_start, seg_end, old_pos, old_pos.clone(), 'hinge')

    # Random angle in [-max_angle, +max_angle]
    angle = (2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0) * cfg.max_angle_hinge

    axis = torch.tensor([dx, dy, dz], dtype=dtype, device=device)
    R = rodrigues_rotation_matrix(axis, angle, dtype, device)

    new_pos = apply_rotation_to_beads_unwrapped(unwrapped, pos_a, R, ns)

    return MoveProposal(chain_idx, seg_start, seg_end, old_pos, new_pos, 'hinge')


def propose_tail_move(state: ChainState, chain_idx: int, seg_start: int,
                       seg_end: int, is_n_terminal: bool, gen: torch.Generator,
                       cfg: SimulationConfig) -> MoveProposal:
    """
    Propose a tail rotation: rotate the N-terminal or C-terminal segment
    around an axis defined by two fixed interior beads.
    """
    N = cfg.N
    ns = state.ns
    device = cfg.get_torch_device()
    dtype = cfg.dtype
    positions = state.positions[chain_idx]
    box = ns.box_size
    inv_box = ns._inv_box

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

    pos_a = positions[a_idx]
    pos_b = positions[b_idx]

    # Axis direction via scalar MIC
    dx, dy, dz = _mic_delta_scalar(pos_a, pos_b, box, inv_box)
    axis_len = math.sqrt(dx * dx + dy * dy + dz * dz)

    if axis_len < AXIS_EPS:
        # Fallback: use orthogonal axis (scalar)
        c_idx = min(a_idx + 1, N - 1)
        tx, ty, tz = _mic_delta_scalar(positions[a_idx], positions[c_idx], box, inv_box)
        t_norm = math.sqrt(tx * tx + ty * ty + tz * tz)
        # Choose reference vector
        if t_norm > AXIS_EPS:
            if abs(tx) / t_norm > 0.9:
                rx, ry, rz = 0.0, 1.0, 0.0
            else:
                rx, ry, rz = 1.0, 0.0, 0.0
            # Cross product
            dx = ty * rz - tz * ry
            dy = tz * rx - tx * rz
            dz = tx * ry - ty * rx
            axis_len = math.sqrt(dx * dx + dy * dy + dz * dz)

        if axis_len < AXIS_EPS:
            old_pos = positions[move_start:move_end].clone()
            mtype = 'n_tail' if is_n_terminal else 'c_tail'
            return MoveProposal(chain_idx, move_start, move_end,
                                old_pos, old_pos.clone(), mtype)

    angle = (2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0) * cfg.max_angle_hinge
    axis = torch.tensor([dx, dy, dz], dtype=dtype, device=device)
    R = rodrigues_rotation_matrix(axis, angle, dtype, device)

    old_pos = positions[move_start:move_end].clone()

    # Sequential unwrap from anchor along chain backbone, then rotate
    # For N-terminal: anchor (a_idx) is after the segment — unwrap backward
    # For C-terminal: anchor (a_idx) is before the segment — unwrap forward
    unwrapped = ns.unwrap_chain_from_anchor(old_pos, pos_a)
    new_pos = apply_rotation_to_beads_unwrapped(unwrapped, pos_a, R, ns)

    mtype = 'n_tail' if is_n_terminal else 'c_tail'
    return MoveProposal(chain_idx, move_start, move_end, old_pos, new_pos, mtype)


def propose_pivot_move(state: ChainState, chain_idx: int,
                        gen: torch.Generator,
                        cfg: SimulationConfig) -> MoveProposal:
    """
    Propose a pivot move: apply uniform random SO(3) rotation to one side
    of the chain around a randomly chosen pivot point.

    Uses fast scalar unwrapping + scalar Rodrigues for minimal overhead.
    """
    N = cfg.N
    ns = state.ns
    device = cfg.get_torch_device()
    dtype = cfg.dtype
    positions = state.positions[chain_idx]  # [N, 3]

    if N <= 2:
        old_pos = positions.clone()
        return MoveProposal(chain_idx, 0, N, old_pos, old_pos.clone(), 'pivot')

    # Random pivot index (exclude endpoints)
    pivot_idx = 1 + int(torch.rand(1, generator=gen, dtype=dtype, device=device).item() * (N - 2))
    pivot_idx = min(pivot_idx, N - 2)

    # Random side: 0 = N-terminal [0, pivot), 1 = C-terminal (pivot, N)
    side = int(torch.rand(1, generator=gen, dtype=dtype, device=device).item() * 2)

    if side == 0:
        rot_start = 0
        rot_end = pivot_idx
    else:
        rot_start = pivot_idx + 1
        rot_end = N

    n_rot = rot_end - rot_start
    if n_rot == 0:
        old_pos = positions[rot_start:rot_end].clone()
        return MoveProposal(chain_idx, rot_start, rot_end, old_pos, old_pos.clone(), 'pivot')

    # Anchor position (pivot bead)
    anchor = positions[pivot_idx].clone()

    # Get beads to rotate in chain order
    if side == 0:
        beads_to_rotate = positions[rot_start:rot_end].flip(0)
    else:
        beads_to_rotate = positions[rot_start:rot_end]

    # Sequential PBC unwrapping (fast scalar version)
    unwrapped = ns.unwrap_chain_from_anchor(beads_to_rotate, anchor)

    # Random SO(3) rotation (fast scalar Rodrigues)
    R = random_so3_matrix(gen, dtype, device)

    # Rotate unwrapped positions around anchor
    relative = unwrapped - anchor.unsqueeze(0)
    rotated = torch.mm(relative, R.t())  # [n_rot, 3]
    new_unwrapped = anchor.unsqueeze(0) + rotated

    # Wrap back into box
    new_pos = ns.wrap(new_unwrapped)

    # If side==0, reverse back to chain order
    if side == 0:
        new_pos = new_pos.flip(0)

    old_pos = positions[rot_start:rot_end].clone()

    return MoveProposal(chain_idx, rot_start, rot_end, old_pos, new_pos, 'pivot')


def propose_batch_pivot_moves(state: 'ChainState', chain_indices: list,
                               rand_pool, cfg: SimulationConfig,
                               ns_work: 'NumberSpace',
                               positions_work: torch.Tensor) -> list:
    """
    Propose B pivot moves using pre-generated random numbers.
    Works on CPU positions (positions_work: [n_chains, N, 3] on CPU).

    Uses scalar unwrapping per proposal (sequential by nature due to PBC
    chaining), but rotation application and wrapping are vectorized.

    Args:
        state: ChainState (used for segment info only)
        chain_indices: list of chain indices to propose pivots for
        rand_pool: RandPool for random numbers
        cfg: SimulationConfig
        ns_work: NumberSpace on the working device (CPU)
        positions_work: [n_chains, N, 3] positions tensor on working device

    Returns:
        list of MoveProposal (on CPU)
    """
    N = cfg.N
    dtype = cfg.dtype
    B = len(chain_indices)
    results = []

    if N <= 2:
        for chain_idx in chain_indices:
            old_pos = positions_work[chain_idx].clone()
            results.append(MoveProposal(chain_idx, 0, N, old_pos, old_pos.clone(), 'pivot'))
        return results

    for chain_idx in chain_indices:
        positions = positions_work[chain_idx]  # [N, 3]

        pivot_idx = 1 + int(rand_pool.next() * (N - 2))
        pivot_idx = min(pivot_idx, N - 2)
        side = int(rand_pool.next() * 2)

        if side == 0:
            rot_start, rot_end = 0, pivot_idx
        else:
            rot_start, rot_end = pivot_idx + 1, N

        n_rot = rot_end - rot_start
        if n_rot == 0:
            old_pos = positions[rot_start:rot_end].clone()
            results.append(MoveProposal(chain_idx, rot_start, rot_end,
                                        old_pos, old_pos.clone(), 'pivot'))
            continue

        anchor = positions[pivot_idx]

        if side == 0:
            beads_to_rotate = positions[rot_start:rot_end].flip(0)
        else:
            beads_to_rotate = positions[rot_start:rot_end]

        # Unwrap on CPU (fast scalar)
        unwrapped = ns_work.unwrap_chain_from_anchor(beads_to_rotate, anchor)

        # SO(3) rotation using pool randoms + scalar Rodrigues
        angle = 2.0 * math.pi * rand_pool.next()
        while True:
            u = 2.0 * rand_pool.next() - 1.0
            v = 2.0 * rand_pool.next() - 1.0
            s = u * u + v * v
            if s < 1.0 and s > 1e-10:
                break
        factor = 2.0 * math.sqrt(1.0 - s)
        ux, uy, uz = u * factor, v * factor, 1.0 - 2.0 * s
        c = math.cos(angle); si = math.sin(angle); t = 1.0 - c
        R = torch.tensor([
            [t*ux*ux+c, t*ux*uy-si*uz, t*ux*uz+si*uy],
            [t*ux*uy+si*uz, t*uy*uy+c, t*uy*uz-si*ux],
            [t*ux*uz-si*uy, t*uy*uz+si*ux, t*uz*uz+c],
        ], dtype=dtype)  # CPU

        relative = unwrapped - anchor.unsqueeze(0)
        rotated = torch.mm(relative, R.t())
        new_unwrapped = anchor.unsqueeze(0) + rotated
        new_pos = ns_work.wrap(new_unwrapped)

        if side == 0:
            new_pos = new_pos.flip(0)

        old_pos = positions[rot_start:rot_end].clone()
        results.append(MoveProposal(chain_idx, rot_start, rot_end, old_pos, new_pos, 'pivot'))

    return results


def propose_batch_pivot_moves_fused(state: 'ChainState', chain_indices: list,
                                     rand_pool, cfg: SimulationConfig,
                                     ns_work: 'NumberSpace',
                                     positions_work: torch.Tensor,
                                     gpu_device: torch.device):
    """
    Propose B pivot moves and pack into a BatchProposal on GPU.

    Proposals generated on CPU (sequential PBC unwrapping required),
    then packed directly into GPU batch tensors — no intermediate
    MoveProposal objects.

    Returns:
        BatchProposal with positions on gpu_device
    """
    from .batch_proposal import BatchProposal

    N = cfg.N
    dtype = cfg.dtype
    B = len(chain_indices)

    # First pass: generate all proposals, track max_moved
    pivot_data = []  # (chain_idx, rot_start, n_rot, old_pos_cpu, new_pos_cpu)

    for chain_idx in chain_indices:
        positions = positions_work[chain_idx]

        if N <= 2:
            old_pos = positions[:N].clone()
            pivot_data.append((chain_idx, 0, N, old_pos, old_pos.clone()))
            continue

        pivot_idx = 1 + int(rand_pool.next() * (N - 2))
        pivot_idx = min(pivot_idx, N - 2)
        side = int(rand_pool.next() * 2)

        if side == 0:
            rot_start, rot_end = 0, pivot_idx
        else:
            rot_start, rot_end = pivot_idx + 1, N

        n_rot = rot_end - rot_start
        if n_rot == 0:
            pivot_data.append((chain_idx, rot_start, 0,
                               positions[rot_start:rot_end].clone(),
                               positions[rot_start:rot_end].clone()))
            continue

        anchor = positions[pivot_idx]
        beads = positions[rot_start:rot_end].flip(0) if side == 0 else positions[rot_start:rot_end]

        unwrapped = ns_work.unwrap_chain_from_anchor(beads, anchor)

        angle = 2.0 * math.pi * rand_pool.next()
        while True:
            u = 2.0 * rand_pool.next() - 1.0
            v = 2.0 * rand_pool.next() - 1.0
            s2 = u * u + v * v
            if s2 < 1.0 and s2 > 1e-10:
                break
        factor = 2.0 * math.sqrt(1.0 - s2)
        ux, uy, uz = u * factor, v * factor, 1.0 - 2.0 * s2
        c = math.cos(angle); si = math.sin(angle); t = 1.0 - c
        R = torch.tensor([
            [t*ux*ux+c, t*ux*uy-si*uz, t*ux*uz+si*uy],
            [t*ux*uy+si*uz, t*uy*uy+c, t*uy*uz-si*ux],
            [t*ux*uz-si*uy, t*uy*uz+si*ux, t*uz*uz+c],
        ], dtype=dtype)

        relative = unwrapped - anchor.unsqueeze(0)
        rotated = torch.mm(relative, R.t())
        new_pos = ns_work.wrap(anchor.unsqueeze(0) + rotated)
        if side == 0:
            new_pos = new_pos.flip(0)

        old_pos = positions[rot_start:rot_end].clone()
        pivot_data.append((chain_idx, rot_start, n_rot, old_pos, new_pos))

    # Pack into BatchProposal on GPU (one bulk transfer)
    max_moved = max(d[2] for d in pivot_data) if pivot_data else 0
    if max_moved == 0:
        max_moved = 1

    bp = BatchProposal(B, max_moved, gpu_device, dtype)
    # Build CPU batch tensors, then transfer once
    old_cpu = torch.zeros(B, max_moved, 3, dtype=dtype)
    new_cpu = torch.zeros(B, max_moved, 3, dtype=dtype)

    for bi, (ci, rs, nr, old_p, new_p) in enumerate(pivot_data):
        bp.chain_idx[bi] = ci
        bp.bead_start[bi] = rs
        bp.n_moved[bi] = nr
        bp.move_types[bi] = 'pivot'
        bp.valid[bi] = nr > 0
        if nr > 0:
            old_cpu[bi, :nr] = old_p
            new_cpu[bi, :nr] = new_p

    # Single CPU→GPU transfer for the entire batch
    bp.old_pos = old_cpu.to(gpu_device)
    bp.new_pos = new_cpu.to(gpu_device)

    return bp


def propose_pivot_move_pooled(positions_cpu: torch.Tensor, chain_idx: int,
                               rand_pool, ns_cpu, cfg,
                               device_out: torch.device) -> MoveProposal:
    """
    Propose pivot move using pre-generated random numbers and CPU positions.
    Zero GPU→CPU syncs. Results placed on device_out.
    """
    N = cfg.N
    dtype = cfg.dtype
    positions = positions_cpu[chain_idx]  # [N, 3] on CPU

    if N <= 2:
        old_pos = positions.clone().to(device_out)
        return MoveProposal(chain_idx, 0, N, old_pos, old_pos.clone(), 'pivot')

    pivot_idx = 1 + int(rand_pool.next() * (N - 2))
    pivot_idx = min(pivot_idx, N - 2)
    side = int(rand_pool.next() * 2)

    if side == 0:
        rot_start, rot_end = 0, pivot_idx
    else:
        rot_start, rot_end = pivot_idx + 1, N

    n_rot = rot_end - rot_start
    if n_rot == 0:
        old_pos = positions[rot_start:rot_end].clone().to(device_out)
        return MoveProposal(chain_idx, rot_start, rot_end, old_pos, old_pos.clone(), 'pivot')

    anchor = positions[pivot_idx]

    if side == 0:
        beads_to_rotate = positions[rot_start:rot_end].flip(0)
    else:
        beads_to_rotate = positions[rot_start:rot_end]

    # Unwrap on CPU (fast scalar)
    unwrapped = ns_cpu.unwrap_chain_from_anchor(beads_to_rotate, anchor)

    # SO(3) rotation using pool randoms + scalar Rodrigues
    angle = 2.0 * math.pi * rand_pool.next()
    while True:
        u = 2.0 * rand_pool.next() - 1.0
        v = 2.0 * rand_pool.next() - 1.0
        s = u * u + v * v
        if s < 1.0 and s > 1e-10:
            break
    factor = 2.0 * math.sqrt(1.0 - s)
    ux, uy, uz = u * factor, v * factor, 1.0 - 2.0 * s
    c = math.cos(angle); si = math.sin(angle); t = 1.0 - c
    R = torch.tensor([
        [t*ux*ux+c, t*ux*uy-si*uz, t*ux*uz+si*uy],
        [t*ux*uy+si*uz, t*uy*uy+c, t*uy*uz-si*ux],
        [t*ux*uz-si*uy, t*uy*uz+si*ux, t*uz*uz+c],
    ], dtype=dtype)  # CPU

    relative = unwrapped - anchor.unsqueeze(0)
    rotated = torch.mm(relative, R.t())
    new_unwrapped = anchor.unsqueeze(0) + rotated
    new_pos = ns_cpu.wrap(new_unwrapped)

    if side == 0:
        new_pos = new_pos.flip(0)

    old_pos = positions[rot_start:rot_end].clone()

    # Move results to target device
    if device_out.type != 'cpu':
        old_pos = old_pos.to(device_out)
        new_pos = new_pos.to(device_out)

    return MoveProposal(chain_idx, rot_start, rot_end, old_pos, new_pos, 'pivot')


def propose_segment_move(state: ChainState, chain_idx: int, local_seg: int,
                          gen: torch.Generator,
                          cfg: SimulationConfig) -> MoveProposal:
    """
    Propose a move for a given segment, choosing the appropriate move type
    based on segment position (inner=hinge, terminal=tail).
    """
    seg_info = state.segments
    seg_type = seg_info.get_segment_type(local_seg)
    seg_start, seg_end = seg_info.get_segment_range(chain_idx, local_seg)

    if seg_type == SegmentInfo.INNER:
        return propose_hinge_move(state, chain_idx, seg_start, seg_end, gen, cfg)
    elif seg_type == SegmentInfo.N_TERMINAL:
        return propose_tail_move(state, chain_idx, seg_start, seg_end, True, gen, cfg)
    elif seg_type == SegmentInfo.C_TERMINAL:
        return propose_tail_move(state, chain_idx, seg_start, seg_end, False, gen, cfg)
    elif seg_type == SegmentInfo.BOTH:
        return propose_tail_move(state, chain_idx, seg_start, seg_end, True, gen, cfg)
    else:
        raise ValueError(f"Unknown segment type: {seg_type}")


# ---------------------------------------------------------------------------
# Batched GPU proposal generation
# ---------------------------------------------------------------------------

def _batched_rodrigues(axes: torch.Tensor, angles: torch.Tensor,
                        valid: torch.Tensor) -> torch.Tensor:
    """
    Build B rotation matrices via batched Rodrigues formula on GPU.

    Args:
        axes: [B, 3] axis vectors (not normalized)
        angles: [B] rotation angles in radians
        valid: [B] bool mask (False → identity matrix)

    Returns:
        [B, 3, 3] rotation matrices
    """
    B = axes.shape[0]
    device = axes.device
    dtype = axes.dtype

    norms = torch.norm(axes, dim=1, keepdim=True).clamp(min=1e-10)
    u = axes / norms  # [B, 3]

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    t = 1.0 - cos_a

    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

    R = torch.zeros(B, 3, 3, dtype=dtype, device=device)
    R[:, 0, 0] = t * ux * ux + cos_a
    R[:, 0, 1] = t * ux * uy - sin_a * uz
    R[:, 0, 2] = t * ux * uz + sin_a * uy
    R[:, 1, 0] = t * ux * uy + sin_a * uz
    R[:, 1, 1] = t * uy * uy + cos_a
    R[:, 1, 2] = t * uy * uz - sin_a * ux
    R[:, 2, 0] = t * ux * uz - sin_a * uy
    R[:, 2, 1] = t * uy * uz + sin_a * ux
    R[:, 2, 2] = t * uz * uz + cos_a

    if not valid.all():
        R[~valid] = torch.eye(3, dtype=dtype, device=device)

    return R


def propose_batch_segment_moves(state: ChainState,
                                 segment_list: list,
                                 gen: torch.Generator,
                                 cfg: SimulationConfig) -> list:
    """
    Propose B segment moves at once using batched GPU operations.

    One kernel launch for rotation matrix construction (batched Rodrigues),
    one kernel for rotation application (batched bmm), one for wrapping.

    Args:
        state: ChainState
        segment_list: list of (chain_idx, local_seg) tuples
        gen: torch.Generator
        cfg: SimulationConfig

    Returns:
        list of MoveProposal
    """
    B = len(segment_list)
    N = cfg.N
    ns = state.ns
    device = cfg.get_torch_device()
    dtype = cfg.dtype
    box = ns.box_size
    inv_box = ns._inv_box
    seg_info = state.segments

    # --- Phase 1: Collect segment metadata (Python, ~0.01ms) ---
    meta = []
    for chain_idx, local_seg in segment_list:
        seg_type = seg_info.get_segment_type(local_seg)
        seg_start, seg_end = seg_info.get_segment_range(chain_idx, local_seg)

        if seg_type == SegmentInfo.INNER:
            a_idx = max(seg_start - 1, 0)
            b_idx = min(seg_end, N - 1)
            move_start, move_end = seg_start, seg_end
            mtype = 'hinge'
        elif seg_type == SegmentInfo.N_TERMINAL or seg_type == SegmentInfo.BOTH:
            a_idx = min(seg_end, N - 1)
            b_idx = min(seg_end + 1, N - 1)
            move_start, move_end = 0, seg_end
            mtype = 'n_tail'
        else:  # C_TERMINAL
            a_idx = max(seg_start - 1, 0)
            b_idx = max(seg_start - 2, 0)
            move_start, move_end = seg_start, N
            mtype = 'c_tail'

        meta.append((chain_idx, a_idx, b_idx, move_start, move_end, mtype))

    # --- Phase 2: Gather axis beads + compute axes (GPU batched) ---
    positions_flat = state.get_all_flat()

    a_global = torch.tensor([m[0] * N + m[1] for m in meta],
                            dtype=torch.long, device=device)
    b_global = torch.tensor([m[0] * N + m[2] for m in meta],
                            dtype=torch.long, device=device)

    pos_a = positions_flat[a_global]  # [B, 3]
    pos_b = positions_flat[b_global]  # [B, 3]

    axes = pos_b - pos_a
    axes = axes - box * torch.round(axes * inv_box)  # [B, 3]
    axis_norms = torch.norm(axes, dim=1)
    valid = axis_norms >= AXIS_EPS

    # Handle degenerate axes with fallback (tail moves)
    degen = ~valid
    if degen.any():
        for bi in range(B):
            if degen[bi]:
                ci, a_idx, b_idx, ms, me, mt = meta[bi]
                c_idx = min(a_idx + 1, N - 1)
                gs_a = ci * N + a_idx
                gs_c = ci * N + c_idx
                tang = positions_flat[gs_c] - positions_flat[gs_a]
                tang = tang - box * torch.round(tang * inv_box)
                t_norm = torch.norm(tang)
                if t_norm > AXIS_EPS:
                    if abs(tang[0].item()) / t_norm.item() > 0.9:
                        ref = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
                    else:
                        ref = torch.tensor([1., 0., 0.], dtype=dtype, device=device)
                    fb_axis = torch.cross(tang, ref)
                    if torch.norm(fb_axis) >= AXIS_EPS:
                        axes[bi] = fb_axis
                        valid[bi] = True

    # --- Phase 3: Batched random angles + rotation matrices ---
    angles = (2.0 * torch.rand(B, generator=gen, dtype=dtype, device=device)
              - 1.0) * cfg.max_angle_hinge
    R_batch = _batched_rodrigues(axes, angles, valid)  # [B, 3, 3]

    # --- Phase 4: Gather old positions + apply batched rotation ---
    max_moved = max(m[4] - m[3] for m in meta)
    n_moved_list = [m[4] - m[3] for m in meta]

    old_batch = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
    for bi, (ci, _, _, ms, me, _) in enumerate(meta):
        nm = me - ms
        old_batch[bi, :nm] = state.positions[ci, ms:me]

    # MIC delta from rotation center (pos_a)
    delta = old_batch - pos_a[:, None, :]  # [B, max_moved, 3]
    delta = delta - box * torch.round(delta * inv_box)

    # Batched rotation: [B, max_moved, 3] @ [B, 3, 3]^T
    rotated = torch.bmm(delta, R_batch.transpose(1, 2))
    new_batch = pos_a[:, None, :] + rotated
    new_batch = ns.wrap(new_batch)

    # --- Phase 5: Create MoveProposal objects ---
    results = []
    for bi, (ci, _, _, ms, me, mtype) in enumerate(meta):
        nm = n_moved_list[bi]
        old_pos = old_batch[bi, :nm].clone()
        new_pos = new_batch[bi, :nm] if valid[bi] else old_pos.clone()
        results.append(MoveProposal(ci, ms, me, old_pos, new_pos, mtype))

    return results


def propose_batch_segment_moves_fused(state: ChainState,
                                       segment_list: list,
                                       gen: torch.Generator,
                                       cfg: SimulationConfig):
    """
    Propose B segment moves returning a BatchProposal (no MoveProposal objects).

    Same GPU-batched algorithm as propose_batch_segment_moves but keeps
    positions as batch tensors throughout, eliminating the unpack/repack
    cycle that costs ~3ms of Python overhead per batch.

    Returns:
        BatchProposal with old_pos, new_pos as [B, max_moved, 3] GPU tensors
    """
    from .batch_proposal import BatchProposal

    B = len(segment_list)
    N = cfg.N
    ns = state.ns
    device = cfg.get_torch_device()
    dtype = cfg.dtype
    box = ns.box_size
    inv_box = ns._inv_box
    seg_info = state.segments

    # Phase 1: Collect segment metadata
    meta = []
    for chain_idx, local_seg in segment_list:
        seg_type = seg_info.get_segment_type(local_seg)
        seg_start, seg_end = seg_info.get_segment_range(chain_idx, local_seg)

        if seg_type == SegmentInfo.INNER:
            a_idx = max(seg_start - 1, 0)
            b_idx = min(seg_end, N - 1)
            move_start, move_end = seg_start, seg_end
            mtype = 'hinge'
        elif seg_type == SegmentInfo.N_TERMINAL or seg_type == SegmentInfo.BOTH:
            a_idx = min(seg_end, N - 1)
            b_idx = min(seg_end + 1, N - 1)
            move_start, move_end = 0, seg_end
            mtype = 'n_tail'
        else:
            a_idx = max(seg_start - 1, 0)
            b_idx = max(seg_start - 2, 0)
            move_start, move_end = seg_start, N
            mtype = 'c_tail'
        meta.append((chain_idx, a_idx, b_idx, move_start, move_end, mtype))

    # Phase 2: GPU batched axes
    positions_flat = state.get_all_flat()
    a_global = torch.tensor([m[0] * N + m[1] for m in meta],
                            dtype=torch.long, device=device)
    b_global = torch.tensor([m[0] * N + m[2] for m in meta],
                            dtype=torch.long, device=device)
    pos_a = positions_flat[a_global]
    pos_b = positions_flat[b_global]
    axes = pos_b - pos_a
    axes = axes - box * torch.round(axes * inv_box)
    axis_norms = torch.norm(axes, dim=1)
    valid_mask = axis_norms >= AXIS_EPS

    degen = ~valid_mask
    if degen.any():
        for bi in range(B):
            if degen[bi]:
                ci, a_idx, b_idx_, ms, me, mt = meta[bi]
                c_idx = min(a_idx + 1, N - 1)
                tang = positions_flat[ci * N + c_idx] - positions_flat[ci * N + a_idx]
                tang = tang - box * torch.round(tang * inv_box)
                t_norm = torch.norm(tang)
                if t_norm > AXIS_EPS:
                    ref_idx = 1 if abs(tang[0].item()) / t_norm.item() > 0.9 else 0
                    ref = torch.zeros(3, dtype=dtype, device=device)
                    ref[ref_idx] = 1.0
                    fb_axis = torch.cross(tang, ref)
                    if torch.norm(fb_axis) >= AXIS_EPS:
                        axes[bi] = fb_axis
                        valid_mask[bi] = True

    # Phase 3: Batched rotation matrices
    angles = (2.0 * torch.rand(B, generator=gen, dtype=dtype, device=device)
              - 1.0) * cfg.max_angle_hinge
    R_batch = _batched_rodrigues(axes, angles, valid_mask)

    # Phase 4: Gather old positions + apply batched rotation
    max_moved = max(m[4] - m[3] for m in meta)

    old_batch = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
    for bi, (ci, _, _, ms, me, _) in enumerate(meta):
        nm = me - ms
        old_batch[bi, :nm] = state.positions[ci, ms:me]

    delta = old_batch - pos_a[:, None, :]
    delta = delta - box * torch.round(delta * inv_box)
    rotated = torch.bmm(delta, R_batch.transpose(1, 2))
    new_batch = pos_a[:, None, :] + rotated
    new_batch = ns.wrap(new_batch)

    # Invalidated proposals get old_pos copied to new_pos
    invalid = ~valid_mask
    if invalid.any():
        new_batch[invalid] = old_batch[invalid]

    # Phase 5: Pack into BatchProposal (NO per-proposal Python loop)
    bp = BatchProposal(B, max_moved, device, dtype)
    bp.old_pos = old_batch    # Already [B, max_moved, 3] on GPU
    bp.new_pos = new_batch    # Already [B, max_moved, 3] on GPU
    for bi, (ci, _, _, ms, me, mtype) in enumerate(meta):
        bp.chain_idx[bi] = ci
        bp.bead_start[bi] = ms
        bp.n_moved[bi] = me - ms
        bp.move_types[bi] = mtype
        bp.valid[bi] = valid_mask[bi].item()

    return bp
