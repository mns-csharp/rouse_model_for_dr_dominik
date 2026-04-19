"""MoveProposer — all MC move proposers (hinge / tail / pivot / segment,
sequential and batched variants). All methods are @staticmethod.
"""

import math
import torch

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.chain.chain_state import ChainState
from rouse_model_python.src.libs.chain.segment_info import SegmentInfo
from rouse_model_python.src.libs.number_space.number_space import NumberSpace
from rouse_model_python.src.libs.mc_moves.move_proposal import MoveProposal
from rouse_model_python.src.libs.mc_moves.batch_proposal import BatchProposal


class MoveProposer:
    AXIS_EPS = 1e-7
    BOND_TOL = 0.05

    @staticmethod
    def random_so3_axis_angle(gen, dtype=torch.float64, device=None):
        while True:
            u1 = 2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0
            v1 = 2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0
            s1 = u1*u1 + v1*v1
            if 1e-10 < s1 < 1.0: break
        while True:
            u2 = 2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0
            v2 = 2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0
            s2 = u2*u2 + v2*v2
            if 1e-10 < s2 < 1.0: break
        q0 = u1; q1 = v1
        factor = math.sqrt((1.0 - s1) / s2)
        q2 = u2 * factor; q3 = v2 * factor
        qnorm = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        q0 /= qnorm; q1 /= qnorm; q2 /= qnorm; q3 /= qnorm
        if q0 < 0.0:
            q0 = -q0; q1 = -q1; q2 = -q2; q3 = -q3
        cos_half = max(-1.0, min(1.0, q0))
        angle = 2.0 * math.acos(cos_half)
        sin_half = math.sqrt(max(0.0, 1.0 - cos_half * cos_half))
        if sin_half < 1e-12:
            return 1.0, 0.0, 0.0, 0.0
        inv = 1.0 / sin_half
        return q1 * inv, q2 * inv, q3 * inv, angle

    @staticmethod
    def max_perp_to_axis_torch(unwrapped: torch.Tensor, anchor: torch.Tensor,
                               ux: float, uy: float, uz: float) -> float:
        rel = unwrapped - anchor.unsqueeze(0)
        axis = torch.tensor([ux, uy, uz], dtype=rel.dtype, device=rel.device)
        proj = (rel * axis).sum(dim=-1, keepdim=True)
        perp = rel - proj * axis
        return float(torch.linalg.vector_norm(perp, dim=-1).max().item())

    @staticmethod
    def cap_angles_batched_by_delta(angles: torch.Tensor, axes: torch.Tensor,
                                    delta: torch.Tensor, pad_mask: torch.Tensor,
                                    l0: float, tol: float) -> torch.Tensor:
        axis_norms = axes.norm(dim=1, keepdim=True).clamp(min=1e-10)
        u = axes / axis_norms
        proj = (delta * u.unsqueeze(1)).sum(dim=-1, keepdim=True)
        perp = delta - proj * u.unsqueeze(1)
        perp_norms = perp.norm(dim=-1)
        perp_norms = torch.where(pad_mask, perp_norms, torch.zeros_like(perp_norms))
        max_perps = perp_norms.max(dim=1).values
        max_disp = l0 * (1.0 + tol)
        s_required = (max_disp / (2.0 * max_perps.clamp(min=1e-12))).clamp(max=1.0)
        s_actual = torch.sin(0.5 * angles.abs())
        theta_cap = 2.0 * torch.asin(s_required)
        return torch.where(s_actual <= s_required, angles, torch.sign(angles) * theta_cap)

    @staticmethod
    def cap_angle_by_displacement(angle: float, max_perp: float,
                                  l0: float, tol: float = 0.05) -> float:
        if max_perp < 1e-12:
            return angle
        max_disp = l0 * (1.0 + tol)
        s_required = max_disp / (2.0 * max_perp)
        if s_required >= 1.0:
            return angle
        s_actual = abs(math.sin(0.5 * angle))
        if s_actual <= s_required:
            return angle
        theta_cap = 2.0 * math.asin(s_required)
        return math.copysign(theta_cap, angle)

    @staticmethod
    def rodrigues_rotation_matrix(axis, angle, dtype=torch.float64, device=None):
        if device is None:
            device = axis.device
        ux = axis[0].item(); uy = axis[1].item(); uz = axis[2].item()
        norm = math.sqrt(ux * ux + uy * uy + uz * uz)
        if norm < MoveProposer.AXIS_EPS:
            return torch.eye(3, dtype=dtype, device=device)
        inv_norm = 1.0 / norm
        ux *= inv_norm; uy *= inv_norm; uz *= inv_norm
        c = math.cos(angle); s = math.sin(angle); t = 1.0 - c
        return torch.tensor([
            [t*ux*ux+c, t*ux*uy-s*uz, t*ux*uz+s*uy],
            [t*ux*uy+s*uz, t*uy*uy+c, t*uy*uz-s*ux],
            [t*ux*uz-s*uy, t*uy*uz+s*ux, t*uz*uz+c],
        ], dtype=dtype, device=device)

    @staticmethod
    def random_so3_matrix(gen, dtype=torch.float64, device=None):
        while True:
            u1 = 2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0
            v1 = 2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0
            s1 = u1*u1 + v1*v1
            if 1e-10 < s1 < 1.0: break
        while True:
            u2 = 2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0
            v2 = 2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0
            s2 = u2*u2 + v2*v2
            if 1e-10 < s2 < 1.0: break
        q0 = u1; q1 = v1
        factor = math.sqrt((1.0 - s1) / s2)
        q2 = u2 * factor; q3 = v2 * factor
        q0q0 = q0*q0; q1q1 = q1*q1; q2q2 = q2*q2; q3q3 = q3*q3
        q0q1 = q0*q1; q0q2 = q0*q2; q0q3 = q0*q3
        q1q2 = q1*q2; q1q3 = q1*q3; q2q3 = q2*q3
        return torch.tensor([
            [q0q0+q1q1-q2q2-q3q3, 2*(q1q2-q0q3), 2*(q1q3+q0q2)],
            [2*(q1q2+q0q3), q0q0-q1q1+q2q2-q3q3, 2*(q2q3-q0q1)],
            [2*(q1q3-q0q2), 2*(q2q3+q0q1), q0q0-q1q1-q2q2+q3q3],
        ], dtype=dtype, device=device)

    @staticmethod
    def apply_rotation_to_beads_unwrapped(unwrapped, center, R, ns: NumberSpace):
        relative = unwrapped - center.unsqueeze(0)
        rotated = torch.mm(relative, R.t())
        new_pos = center.unsqueeze(0) + rotated
        return ns.wrap(new_pos)

    @staticmethod
    def _mic_delta_scalar(p1, p2, box, inv_box):
        ax, ay, az = p1[0].item(), p1[1].item(), p1[2].item()
        bx, by, bz = p2[0].item(), p2[1].item(), p2[2].item()
        dx = bx - ax; dy = by - ay; dz = bz - az
        dx -= box * round(dx * inv_box)
        dy -= box * round(dy * inv_box)
        dz -= box * round(dz * inv_box)
        return dx, dy, dz

    @staticmethod
    def propose_hinge_move(state: ChainState, chain_idx, seg_start, seg_end,
                           gen, cfg: SimulationConfig) -> MoveProposal:
        N = cfg.N
        ns = state.ns
        device = cfg.get_torch_device()
        dtype = cfg.dtype
        positions = state.positions[chain_idx]
        axis_bead_a = max(seg_start - 1, 0)
        axis_bead_b = min(seg_end, N - 1)
        pos_a = positions[axis_bead_a]
        old_pos = positions[seg_start:seg_end].clone()
        unwrapped = ns.unwrap_chain_from_anchor(old_pos, pos_a)
        box = ns.box_size; inv_box = ns._inv_box
        dx_b, dy_b, dz_b = MoveProposer._mic_delta_scalar(
            positions[seg_end - 1], positions[axis_bead_b], box, inv_box)
        uw_b_x = unwrapped[-1, 0].item() + dx_b
        uw_b_y = unwrapped[-1, 1].item() + dy_b
        uw_b_z = unwrapped[-1, 2].item() + dz_b
        dx = uw_b_x - pos_a[0].item()
        dy = uw_b_y - pos_a[1].item()
        dz = uw_b_z - pos_a[2].item()
        axis_len = math.sqrt(dx*dx + dy*dy + dz*dz)
        if axis_len < MoveProposer.AXIS_EPS:
            return MoveProposal(chain_idx, seg_start, seg_end, old_pos, old_pos.clone(), 'hinge')
        angle = (2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0) * cfg.max_angle_hinge
        inv_len = 1.0 / axis_len
        ux, uy, uz = dx * inv_len, dy * inv_len, dz * inv_len
        max_perp = MoveProposer.max_perp_to_axis_torch(unwrapped, pos_a, ux, uy, uz)
        angle = MoveProposer.cap_angle_by_displacement(
            angle, max_perp, cfg.l0, MoveProposer.BOND_TOL)
        axis = torch.tensor([dx, dy, dz], dtype=dtype, device=device)
        R = MoveProposer.rodrigues_rotation_matrix(axis, angle, dtype, device)
        new_pos = MoveProposer.apply_rotation_to_beads_unwrapped(unwrapped, pos_a, R, ns)
        return MoveProposal(chain_idx, seg_start, seg_end, old_pos, new_pos, 'hinge')

    @staticmethod
    def propose_tail_move(state: ChainState, chain_idx, seg_start, seg_end,
                          is_n_terminal, gen, cfg: SimulationConfig) -> MoveProposal:
        N = cfg.N
        ns = state.ns
        device = cfg.get_torch_device()
        dtype = cfg.dtype
        positions = state.positions[chain_idx]
        box = ns.box_size; inv_box = ns._inv_box
        if is_n_terminal:
            move_start, move_end = 0, seg_end
            a_idx = min(seg_end, N - 1)
            b_idx = min(seg_end + 1, N - 1)
        else:
            move_start, move_end = seg_start, N
            a_idx = max(seg_start - 1, 0)
            b_idx = max(seg_start - 2, 0)
        pos_a = positions[a_idx]
        pos_b = positions[b_idx]
        dx, dy, dz = MoveProposer._mic_delta_scalar(pos_a, pos_b, box, inv_box)
        axis_len = math.sqrt(dx*dx + dy*dy + dz*dz)
        if axis_len < MoveProposer.AXIS_EPS:
            c_idx = min(a_idx + 1, N - 1)
            tx, ty, tz = MoveProposer._mic_delta_scalar(
                positions[a_idx], positions[c_idx], box, inv_box)
            t_norm = math.sqrt(tx*tx + ty*ty + tz*tz)
            if t_norm > MoveProposer.AXIS_EPS:
                if abs(tx) / t_norm > 0.9:
                    rx, ry, rz = 0.0, 1.0, 0.0
                else:
                    rx, ry, rz = 1.0, 0.0, 0.0
                dx = ty*rz - tz*ry
                dy = tz*rx - tx*rz
                dz = tx*ry - ty*rx
                axis_len = math.sqrt(dx*dx + dy*dy + dz*dz)
            if axis_len < MoveProposer.AXIS_EPS:
                old_pos = positions[move_start:move_end].clone()
                mtype = 'n_tail' if is_n_terminal else 'c_tail'
                return MoveProposal(chain_idx, move_start, move_end,
                                    old_pos, old_pos.clone(), mtype)
        angle = (2.0 * torch.rand(1, generator=gen, dtype=dtype, device=device).item() - 1.0) * cfg.max_angle_hinge
        old_pos = positions[move_start:move_end].clone()
        if is_n_terminal:
            unwrapped = ns.unwrap_chain_from_anchor(old_pos.flip(0), pos_a).flip(0)
        else:
            unwrapped = ns.unwrap_chain_from_anchor(old_pos, pos_a)
        inv_len = 1.0 / axis_len
        ux, uy, uz = dx * inv_len, dy * inv_len, dz * inv_len
        max_perp = MoveProposer.max_perp_to_axis_torch(unwrapped, pos_a, ux, uy, uz)
        angle = MoveProposer.cap_angle_by_displacement(
            angle, max_perp, cfg.l0, MoveProposer.BOND_TOL)
        axis = torch.tensor([dx, dy, dz], dtype=dtype, device=device)
        R = MoveProposer.rodrigues_rotation_matrix(axis, angle, dtype, device)
        new_pos = MoveProposer.apply_rotation_to_beads_unwrapped(unwrapped, pos_a, R, ns)
        mtype = 'n_tail' if is_n_terminal else 'c_tail'
        return MoveProposal(chain_idx, move_start, move_end, old_pos, new_pos, mtype)

    @staticmethod
    def propose_pivot_move(state: ChainState, chain_idx, gen,
                           cfg: SimulationConfig) -> MoveProposal:
        N = cfg.N
        ns = state.ns
        device = cfg.get_torch_device()
        dtype = cfg.dtype
        positions = state.positions[chain_idx]
        if N <= 2:
            old_pos = positions.clone()
            return MoveProposal(chain_idx, 0, N, old_pos, old_pos.clone(), 'pivot')
        pivot_idx = 1 + int(torch.rand(1, generator=gen, dtype=dtype, device=device).item() * (N - 2))
        pivot_idx = min(pivot_idx, N - 2)
        side = int(torch.rand(1, generator=gen, dtype=dtype, device=device).item() * 2)
        if side == 0:
            rot_start, rot_end = 0, pivot_idx
        else:
            rot_start, rot_end = pivot_idx + 1, N
        n_rot = rot_end - rot_start
        if n_rot == 0:
            old_pos = positions[rot_start:rot_end].clone()
            return MoveProposal(chain_idx, rot_start, rot_end, old_pos, old_pos.clone(), 'pivot')
        anchor = positions[pivot_idx].clone()
        beads_to_rotate = positions[rot_start:rot_end].flip(0) if side == 0 else positions[rot_start:rot_end]
        unwrapped = ns.unwrap_chain_from_anchor(beads_to_rotate, anchor)
        ux, uy, uz, angle = MoveProposer.random_so3_axis_angle(gen, dtype, device)
        max_perp = MoveProposer.max_perp_to_axis_torch(unwrapped, anchor, ux, uy, uz)
        angle = MoveProposer.cap_angle_by_displacement(
            angle, max_perp, cfg.l0, MoveProposer.BOND_TOL)
        axis = torch.tensor([ux, uy, uz], dtype=dtype, device=device)
        R = MoveProposer.rodrigues_rotation_matrix(axis, angle, dtype, device)
        relative = unwrapped - anchor.unsqueeze(0)
        rotated = torch.mm(relative, R.t())
        new_unwrapped = anchor.unsqueeze(0) + rotated
        new_pos = ns.wrap(new_unwrapped)
        if side == 0:
            new_pos = new_pos.flip(0)
        old_pos = positions[rot_start:rot_end].clone()
        return MoveProposal(chain_idx, rot_start, rot_end, old_pos, new_pos, 'pivot')

    @staticmethod
    def propose_batch_pivot_moves(state, chain_indices, rand_pool, cfg,
                                  ns_work, positions_work):
        N = cfg.N
        dtype = cfg.dtype
        results = []
        if N <= 2:
            for chain_idx in chain_indices:
                old_pos = positions_work[chain_idx].clone()
                results.append(MoveProposal(chain_idx, 0, N, old_pos, old_pos.clone(), 'pivot'))
            return results
        for chain_idx in chain_indices:
            positions = positions_work[chain_idx]
            pivot_idx = min(1 + int(rand_pool.next() * (N - 2)), N - 2)
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
            beads_to_rotate = positions[rot_start:rot_end].flip(0) if side == 0 else positions[rot_start:rot_end]
            unwrapped = ns_work.unwrap_chain_from_anchor(beads_to_rotate, anchor)
            angle = 2.0 * math.pi * rand_pool.next()
            while True:
                u = 2.0 * rand_pool.next() - 1.0
                v = 2.0 * rand_pool.next() - 1.0
                s = u*u + v*v
                if 1e-10 < s < 1.0: break
            factor = 2.0 * math.sqrt(1.0 - s)
            ux, uy, uz = u*factor, v*factor, 1.0 - 2.0*s
            max_perp = MoveProposer.max_perp_to_axis_torch(unwrapped, anchor, ux, uy, uz)
            angle = MoveProposer.cap_angle_by_displacement(
                angle, max_perp, cfg.l0, MoveProposer.BOND_TOL)
            c = math.cos(angle); si = math.sin(angle); t = 1.0 - c
            R = torch.tensor([
                [t*ux*ux+c, t*ux*uy-si*uz, t*ux*uz+si*uy],
                [t*ux*uy+si*uz, t*uy*uy+c, t*uy*uz-si*ux],
                [t*ux*uz-si*uy, t*uy*uz+si*ux, t*uz*uz+c],
            ], dtype=dtype)
            relative = unwrapped - anchor.unsqueeze(0)
            rotated = torch.mm(relative, R.t())
            new_unwrapped = anchor.unsqueeze(0) + rotated
            new_pos = ns_work.wrap(new_unwrapped)
            if side == 0:
                new_pos = new_pos.flip(0)
            old_pos = positions[rot_start:rot_end].clone()
            results.append(MoveProposal(chain_idx, rot_start, rot_end, old_pos, new_pos, 'pivot'))
        return results

    @staticmethod
    def propose_batch_pivot_moves_fused(state, chain_indices, rand_pool, cfg,
                                        ns_work, positions_work, gpu_device):
        N = cfg.N
        dtype = cfg.dtype
        B = len(chain_indices)
        pivot_data = []
        for chain_idx in chain_indices:
            positions = positions_work[chain_idx]
            if N <= 2:
                old_pos = positions[:N].clone()
                pivot_data.append((chain_idx, 0, N, old_pos, old_pos.clone()))
                continue
            pivot_idx = min(1 + int(rand_pool.next() * (N - 2)), N - 2)
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
                s2 = u*u + v*v
                if 1e-10 < s2 < 1.0: break
            factor = 2.0 * math.sqrt(1.0 - s2)
            ux, uy, uz = u*factor, v*factor, 1.0 - 2.0*s2
            max_perp = MoveProposer.max_perp_to_axis_torch(unwrapped, anchor, ux, uy, uz)
            angle = MoveProposer.cap_angle_by_displacement(
                angle, max_perp, cfg.l0, MoveProposer.BOND_TOL)
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
        max_moved = max(d[2] for d in pivot_data) if pivot_data else 0
        if max_moved == 0:
            max_moved = 1
        bp = BatchProposal(B, max_moved, gpu_device, dtype)
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
        bp.old_pos = old_cpu.to(gpu_device)
        bp.new_pos = new_cpu.to(gpu_device)
        return bp

    @staticmethod
    def _quat_to_matrix_batched(q: torch.Tensor) -> torch.Tensor:
        """Vectorised quaternion to rotation matrix.

        Input: q of shape (B, 4), unit quaternions (q0 is scalar part).
        Output: R of shape (B, 3, 3). Same algebra as `random_so3_matrix`
        lines 51-58 (single-quat scalar version).
        """
        q0 = q[:, 0]; q1 = q[:, 1]; q2 = q[:, 2]; q3 = q[:, 3]
        q0q0 = q0 * q0; q1q1 = q1 * q1; q2q2 = q2 * q2; q3q3 = q3 * q3
        q0q1 = q0 * q1; q0q2 = q0 * q2; q0q3 = q0 * q3
        q1q2 = q1 * q2; q1q3 = q1 * q3; q2q3 = q2 * q3
        B = q.shape[0]
        R = torch.empty(B, 3, 3, dtype=q.dtype, device=q.device)
        R[:, 0, 0] = q0q0 + q1q1 - q2q2 - q3q3
        R[:, 0, 1] = 2.0 * (q1q2 - q0q3)
        R[:, 0, 2] = 2.0 * (q1q3 + q0q2)
        R[:, 1, 0] = 2.0 * (q1q2 + q0q3)
        R[:, 1, 1] = q0q0 - q1q1 + q2q2 - q3q3
        R[:, 1, 2] = 2.0 * (q2q3 - q0q1)
        R[:, 2, 0] = 2.0 * (q1q3 - q0q2)
        R[:, 2, 1] = 2.0 * (q2q3 + q0q1)
        R[:, 2, 2] = q0q0 - q1q1 - q2q2 + q3q3
        return R

    @staticmethod
    def propose_batch_pivot_moves_vec(state, chain_indices, rand_pool, cfg,
                                       ns_work, positions_work, gpu_device):
        """Vectorised replacement for `propose_batch_pivot_moves_fused`.

        Samples pivot_idx, side, and SO(3) quaternions in batched tensor ops.
        Keeps per-chain unwrap_chain_from_anchor (cheapest path per audit).
        Applies rotation in one bmm + one ns.wrap. Same BatchProposal output
        contract.
        """
        N = cfg.N
        dtype = cfg.dtype
        B = len(chain_indices)
        cpu = torch.device('cpu')
        gen = rand_pool._gen

        if N <= 2:
            max_moved = max(N, 1)
            bp = BatchProposal(B, max_moved, gpu_device, dtype)
            old_cpu = torch.zeros(B, max_moved, 3, dtype=dtype)
            for bi, ci in enumerate(chain_indices):
                old_cpu[bi, :N] = positions_work[ci]
                bp.chain_idx[bi] = ci
                bp.bead_start[bi] = 0
                bp.n_moved[bi] = N
                bp.move_types[bi] = 'pivot'
                bp.valid[bi] = False
            bp.old_pos = old_cpu.to(gpu_device)
            bp.new_pos = old_cpu.clone().to(gpu_device)
            return bp

        gen_device = gen.device
        r = torch.rand(B, 2, generator=gen, dtype=dtype, device=gen_device).to(cpu)
        pivot_idx = (1 + (r[:, 0] * (N - 2)).long()).clamp(max=N - 2)
        side = (r[:, 1] * 2).long().clamp(max=1)
        zeros_B = torch.zeros(B, dtype=torch.long)
        N_fill = torch.full((B,), N, dtype=torch.long)
        rot_start = torch.where(side == 0, zeros_B, pivot_idx + 1)
        rot_end = torch.where(side == 0, pivot_idx, N_fill)
        n_rot = rot_end - rot_start

        n_rot_list = n_rot.tolist()
        pivot_idx_list = pivot_idx.tolist()
        side_list = side.tolist()
        rot_start_list = rot_start.tolist()
        rot_end_list = rot_end.tolist()
        max_moved = max(max(n_rot_list), 1)

        # Match the fused method: axis uniform on S² via Marsaglia, angle uniform on [0, 2π]
        K = 3 * B
        while True:
            uv = 2.0 * torch.rand(K, 2, generator=gen, dtype=dtype, device=gen_device) - 1.0
            s2 = (uv * uv).sum(dim=1)
            valid_mask = (s2 > 1e-10) & (s2 < 1.0)
            if int(valid_mask.sum().item()) >= B:
                break
            K *= 2
        idx = valid_mask.nonzero(as_tuple=True)[0][:B]
        uv_s = uv[idx].to(cpu)
        s2_s = s2[idx].to(cpu)
        factor = 2.0 * (1.0 - s2_s).sqrt()
        axes = torch.empty(B, 3, dtype=dtype)
        axes[:, 0] = uv_s[:, 0] * factor
        axes[:, 1] = uv_s[:, 1] * factor
        axes[:, 2] = 1.0 - 2.0 * s2_s
        angles = (2.0 * math.pi * torch.rand(B, generator=gen, dtype=dtype, device=gen_device)).to(cpu)

        old_cpu = torch.zeros(B, max_moved, 3, dtype=dtype)
        delta_cpu = torch.zeros(B, max_moved, 3, dtype=dtype)
        anchors = torch.zeros(B, 3, dtype=dtype)
        valid_list = [False] * B
        pad_mask = torch.zeros(B, max_moved, dtype=torch.bool)

        for bi in range(B):
            ci = chain_indices[bi]
            rs = rot_start_list[bi]
            re = rot_end_list[bi]
            nr = n_rot_list[bi]
            if nr == 0:
                continue
            s = side_list[bi]
            positions = positions_work[ci]
            anchor = positions[pivot_idx_list[bi]]
            anchors[bi] = anchor
            beads = positions[rs:re].flip(0) if s == 0 else positions[rs:re]
            unwrapped = ns_work.unwrap_chain_from_anchor(beads, anchor)
            if s == 0:
                unwrapped = unwrapped.flip(0)
            old_cpu[bi, :nr] = positions[rs:re]
            delta_cpu[bi, :nr] = unwrapped - anchor.unsqueeze(0)
            valid_list[bi] = True
            pad_mask[bi, :nr] = True

        angles = MoveProposer.cap_angles_batched_by_delta(
            angles, axes, delta_cpu, pad_mask, cfg.l0, MoveProposer.BOND_TOL)
        R = MoveProposer._batched_rodrigues(
            axes, angles, torch.ones(B, dtype=torch.bool))

        rotated = torch.bmm(delta_cpu, R.transpose(1, 2))
        new_cpu = ns_work.wrap(anchors.unsqueeze(1) + rotated)

        if not all(valid_list):
            valid_t = torch.tensor(valid_list, dtype=torch.bool)
            new_cpu[~valid_t] = old_cpu[~valid_t]

        bp = BatchProposal(B, max_moved, gpu_device, dtype)
        bp.old_pos = old_cpu.to(gpu_device)
        bp.new_pos = new_cpu.to(gpu_device)
        for bi in range(B):
            bp.chain_idx[bi] = chain_indices[bi]
            bp.bead_start[bi] = rot_start_list[bi]
            bp.n_moved[bi] = n_rot_list[bi]
            bp.move_types[bi] = 'pivot'
            bp.valid[bi] = valid_list[bi]
        return bp

    @staticmethod
    def propose_batch_pivot_moves_gpu(state, chain_indices_t: torch.Tensor,
                                      gen: torch.Generator,
                                      cfg: SimulationConfig) -> BatchProposal:
        """Fully-vectorised pivot-move proposer that lives on the GPU.

        No CPU mirror, no per-chain Python loop, no per-chain `.to(gpu)`.
        Same physics as `propose_batch_pivot_moves_fused`; RNG consumption
        order DIFFERS (vectorised draws), so this is stat-equivalent rather
        than bit-identical.

        Args:
          state:             ChainState — positions read directly off device.
          chain_indices_t:   int64[B] CUDA tensor of chain indices to pivot.
          gen:               torch.Generator on the same device as state.positions.
          cfg:               SimulationConfig.

        Returns:
          BatchProposal with old_pos / new_pos on device, plus
          chain_idx_t / bead_start_t / n_moved_t populated so multistep_mc.py
          can scatter without per-batch H2D conversion.
        """
        N = cfg.N
        dtype = cfg.dtype
        ns = state.ns
        device = state.positions.device
        B = int(chain_indices_t.shape[0])

        # Degenerate: chains too short to pivot — return a no-op batch.
        if N <= 2 or B == 0:
            max_moved = max(N, 1)
            bp = BatchProposal(B, max_moved, device, dtype)
            old_pos = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
            if B > 0:
                old_pos[:, :N] = state.positions[chain_indices_t]
            bp.old_pos = old_pos
            bp.new_pos = old_pos.clone()
            zero_B = torch.zeros(B, dtype=torch.int64, device=device)
            bp.chain_idx_t = chain_indices_t
            bp.bead_start_t = zero_B
            bp.n_moved_t = torch.full((B,), N, dtype=torch.int64, device=device) if B > 0 else zero_B
            bp.chain_idx = chain_indices_t.tolist()
            bp.bead_start = [0] * B
            bp.n_moved = [N] * B
            bp.move_types = ['pivot'] * B
            bp.valid = [False] * B
            return bp

        # 1. Sample pivot_idx, side, rot_start/rot_end/n_rot — all on device.
        r = torch.rand(B, 2, generator=gen, dtype=dtype, device=device)
        pivot_idx = (1 + (r[:, 0] * (N - 2)).long()).clamp(max=N - 2)
        side = (r[:, 1] * 2).long().clamp(max=1)
        zeros_B = torch.zeros(B, dtype=torch.int64, device=device)
        full_N_B = torch.full((B,), N, dtype=torch.int64, device=device)
        rot_start = torch.where(side == 0, zeros_B, pivot_idx + 1)
        rot_end = torch.where(side == 0, pivot_idx, full_N_B)
        n_rot = rot_end - rot_start  # always >= 1 for pivot_idx in [1, N-2]

        # 2. Pad-to-max_moved strategy.
        if cfg.pivot_pad_strategy == "fixed":
            max_moved = N - 1
        else:
            max_moved = int(n_rot.max().item())  # one small sync per batch
        max_moved = max(max_moved, 1)

        # 3. Vectorised gather of segment positions.
        k_range = torch.arange(max_moved, device=device, dtype=torch.int64)
        valid_lane = k_range.unsqueeze(0) < n_rot.unsqueeze(1)  # [B, M]
        chain_exp = chain_indices_t.unsqueeze(1).expand(B, max_moved)
        bead_idx = (rot_start.unsqueeze(1) + k_range.unsqueeze(0)).clamp(max=N - 1)
        old_pos = state.positions[chain_exp, bead_idx]  # [B, M, 3], natural order
        # For side=0 we unwrap from the flipped (anchor-adjacent first) order.
        bead_idx_flip = (rot_end - 1).unsqueeze(1) - k_range.unsqueeze(0)
        bead_idx_flip = bead_idx_flip.clamp(min=0)
        beads_flipped = state.positions[chain_exp, bead_idx_flip]
        side_mask3d = (side == 0).view(B, 1, 1)
        beads = torch.where(side_mask3d, beads_flipped, old_pos)
        anchors = state.positions[chain_indices_t, pivot_idx]  # [B, 3]

        # 4. Vectorised cumulative-MIC unwrap.
        unwrapped = ns.unwrap_segments_from_anchor(beads, anchors, valid_lane)

        # 5. SO(3) sampling on device (Marsaglia, oversampled).
        K = 4 * B
        uv = 2.0 * torch.rand(K, 2, generator=gen, dtype=dtype, device=device) - 1.0
        s2 = (uv * uv).sum(dim=1)
        ok = (s2 > 1e-10) & (s2 < 1.0)
        idx = ok.nonzero(as_tuple=True)[0][:B]
        # Defensive fallback: if oversampling didn't yield enough valid samples
        # (astronomically rare for K=4B with acceptance ~pi/4), pad with a top-up
        # round. Avoids out-of-bounds slicing that would silently truncate B.
        while idx.numel() < B:
            extra = 4 * (B - int(idx.numel()))
            uv2 = 2.0 * torch.rand(extra, 2, generator=gen, dtype=dtype, device=device) - 1.0
            s2_2 = (uv2 * uv2).sum(dim=1)
            ok2 = (s2_2 > 1e-10) & (s2_2 < 1.0)
            uv = torch.cat([uv[idx], uv2[ok2]], dim=0)
            s2 = (uv * uv).sum(dim=1)
            ok = (s2 > 1e-10) & (s2 < 1.0)
            idx = ok.nonzero(as_tuple=True)[0][:B]
        uv_s = uv[idx]
        s2_s = s2[idx]
        factor = 2.0 * torch.sqrt(1.0 - s2_s)
        axes = torch.stack([uv_s[:, 0] * factor,
                            uv_s[:, 1] * factor,
                            1.0 - 2.0 * s2_s], dim=1)
        angles = 2.0 * math.pi * torch.rand(B, generator=gen, dtype=dtype, device=device)

        # 6. Apply rotation, wrap, undo flip for side=0.
        delta = unwrapped - anchors.unsqueeze(1)
        angles = MoveProposer.cap_angles_batched_by_delta(
            angles, axes, delta, valid_lane, cfg.l0, MoveProposer.BOND_TOL)
        R = MoveProposer._batched_rodrigues(
            axes, angles, torch.ones(B, dtype=torch.bool, device=device))
        rotated = torch.bmm(delta, R.transpose(1, 2))
        new_pos_natural_or_flipped = ns.wrap(anchors.unsqueeze(1) + rotated)
        flip_k = (n_rot.unsqueeze(1) - 1 - k_range.unsqueeze(0)).clamp(min=0)
        new_pos_unflipped = torch.gather(
            new_pos_natural_or_flipped, 1,
            flip_k.unsqueeze(-1).expand(B, max_moved, 3))
        new_pos = torch.where(side_mask3d, new_pos_unflipped, new_pos_natural_or_flipped)
        # Padding lanes: hold identity (= old_pos) so EMM / accept see no garbage.
        new_pos = torch.where(valid_lane.unsqueeze(-1), new_pos, old_pos)

        # 7. Build BatchProposal.
        bp = BatchProposal(B, max_moved, device, dtype)
        bp.old_pos = old_pos
        bp.new_pos = new_pos
        bp.chain_idx_t = chain_indices_t
        bp.bead_start_t = rot_start
        bp.n_moved_t = n_rot
        bp.chain_idx = chain_indices_t.tolist()
        bp.bead_start = rot_start.tolist()
        bp.n_moved = n_rot.tolist()
        bp.move_types = ['pivot'] * B
        bp.valid = [n > 0 for n in bp.n_moved]
        return bp

    @staticmethod
    def propose_pivot_move_pooled(positions_cpu, chain_idx, rand_pool, ns_cpu,
                                  cfg, device_out) -> MoveProposal:
        N = cfg.N
        dtype = cfg.dtype
        positions = positions_cpu[chain_idx]
        if N <= 2:
            old_pos = positions.clone().to(device_out)
            return MoveProposal(chain_idx, 0, N, old_pos, old_pos.clone(), 'pivot')
        pivot_idx = min(1 + int(rand_pool.next() * (N - 2)), N - 2)
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
        beads_to_rotate = positions[rot_start:rot_end].flip(0) if side == 0 else positions[rot_start:rot_end]
        unwrapped = ns_cpu.unwrap_chain_from_anchor(beads_to_rotate, anchor)
        angle = 2.0 * math.pi * rand_pool.next()
        while True:
            u = 2.0 * rand_pool.next() - 1.0
            v = 2.0 * rand_pool.next() - 1.0
            s = u*u + v*v
            if 1e-10 < s < 1.0: break
        factor = 2.0 * math.sqrt(1.0 - s)
        ux, uy, uz = u*factor, v*factor, 1.0 - 2.0*s
        max_perp = MoveProposer.max_perp_to_axis_torch(unwrapped, anchor, ux, uy, uz)
        angle = MoveProposer.cap_angle_by_displacement(
            angle, max_perp, cfg.l0, MoveProposer.BOND_TOL)
        c = math.cos(angle); si = math.sin(angle); t = 1.0 - c
        R = torch.tensor([
            [t*ux*ux+c, t*ux*uy-si*uz, t*ux*uz+si*uy],
            [t*ux*uy+si*uz, t*uy*uy+c, t*uy*uz-si*ux],
            [t*ux*uz-si*uy, t*uy*uz+si*ux, t*uz*uz+c],
        ], dtype=dtype)
        relative = unwrapped - anchor.unsqueeze(0)
        rotated = torch.mm(relative, R.t())
        new_unwrapped = anchor.unsqueeze(0) + rotated
        new_pos = ns_cpu.wrap(new_unwrapped)
        if side == 0:
            new_pos = new_pos.flip(0)
        old_pos = positions[rot_start:rot_end].clone()
        if device_out.type != 'cpu':
            old_pos = old_pos.to(device_out)
            new_pos = new_pos.to(device_out)
        return MoveProposal(chain_idx, rot_start, rot_end, old_pos, new_pos, 'pivot')

    @staticmethod
    def propose_segment_move(state: ChainState, chain_idx, local_seg,
                             gen, cfg: SimulationConfig) -> MoveProposal:
        seg_info = state.segments
        seg_type = seg_info.get_segment_type(local_seg)
        seg_start, seg_end = seg_info.get_segment_range(chain_idx, local_seg)
        if seg_type == SegmentInfo.INNER:
            return MoveProposer.propose_hinge_move(state, chain_idx, seg_start, seg_end, gen, cfg)
        if seg_type == SegmentInfo.N_TERMINAL:
            return MoveProposer.propose_tail_move(state, chain_idx, seg_start, seg_end, True, gen, cfg)
        if seg_type == SegmentInfo.C_TERMINAL:
            return MoveProposer.propose_tail_move(state, chain_idx, seg_start, seg_end, False, gen, cfg)
        if seg_type == SegmentInfo.BOTH:
            return MoveProposer.propose_tail_move(state, chain_idx, seg_start, seg_end, True, gen, cfg)
        raise ValueError(f"Unknown segment type: {seg_type}")

    @staticmethod
    def _batched_rodrigues(axes, angles, valid):
        B = axes.shape[0]
        device = axes.device
        dtype = axes.dtype
        norms = torch.norm(axes, dim=1, keepdim=True).clamp(min=1e-10)
        u = axes / norms
        cos_a = torch.cos(angles); sin_a = torch.sin(angles); t = 1.0 - cos_a
        ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
        R = torch.zeros(B, 3, 3, dtype=dtype, device=device)
        R[:, 0, 0] = t*ux*ux + cos_a
        R[:, 0, 1] = t*ux*uy - sin_a*uz
        R[:, 0, 2] = t*ux*uz + sin_a*uy
        R[:, 1, 0] = t*ux*uy + sin_a*uz
        R[:, 1, 1] = t*uy*uy + cos_a
        R[:, 1, 2] = t*uy*uz - sin_a*ux
        R[:, 2, 0] = t*ux*uz - sin_a*uy
        R[:, 2, 1] = t*uy*uz + sin_a*ux
        R[:, 2, 2] = t*uz*uz + cos_a
        if not valid.all():
            R[~valid] = torch.eye(3, dtype=dtype, device=device)
        return R

    @staticmethod
    def propose_batch_segment_moves(state: ChainState, segment_list,
                                    gen, cfg: SimulationConfig):
        B = len(segment_list)
        N = cfg.N
        ns = state.ns
        device = cfg.get_torch_device()
        dtype = cfg.dtype
        box = ns.box_size; inv_box = ns._inv_box
        seg_info = state.segments
        meta = []
        for chain_idx, local_seg in segment_list:
            seg_type = seg_info.get_segment_type(local_seg)
            seg_start, seg_end = seg_info.get_segment_range(chain_idx, local_seg)
            if seg_type == SegmentInfo.INNER:
                a_idx = max(seg_start - 1, 0); b_idx = min(seg_end, N - 1)
                ms, me = seg_start, seg_end; mtype = 'hinge'
            elif seg_type in (SegmentInfo.N_TERMINAL, SegmentInfo.BOTH):
                a_idx = min(seg_end, N - 1); b_idx = min(seg_end + 1, N - 1)
                ms, me = 0, seg_end; mtype = 'n_tail'
            else:
                a_idx = max(seg_start - 1, 0); b_idx = max(seg_start - 2, 0)
                ms, me = seg_start, N; mtype = 'c_tail'
            meta.append((chain_idx, a_idx, b_idx, ms, me, mtype))
        positions_flat = state.get_all_flat()
        a_global = torch.tensor([m[0]*N + m[1] for m in meta], dtype=torch.long, device=device)
        b_global = torch.tensor([m[0]*N + m[2] for m in meta], dtype=torch.long, device=device)
        pos_a = positions_flat[a_global]
        pos_b = positions_flat[b_global]
        axes = pos_b - pos_a
        axes = axes - box * torch.round(axes * inv_box)
        axis_norms = torch.norm(axes, dim=1)
        valid = axis_norms >= MoveProposer.AXIS_EPS
        degen = ~valid
        if degen.any():
            for bi in range(B):
                if degen[bi]:
                    ci, a_idx, b_idx, ms, me, mt = meta[bi]
                    c_idx = min(a_idx + 1, N - 1)
                    gs_a = ci*N + a_idx; gs_c = ci*N + c_idx
                    tang = positions_flat[gs_c] - positions_flat[gs_a]
                    tang = tang - box * torch.round(tang * inv_box)
                    t_norm = torch.norm(tang)
                    if t_norm > MoveProposer.AXIS_EPS:
                        if abs(tang[0].item()) / t_norm.item() > 0.9:
                            ref = torch.tensor([0., 1., 0.], dtype=dtype, device=device)
                        else:
                            ref = torch.tensor([1., 0., 0.], dtype=dtype, device=device)
                        fb_axis = torch.cross(tang, ref)
                        if torch.norm(fb_axis) >= MoveProposer.AXIS_EPS:
                            axes[bi] = fb_axis
                            valid[bi] = True
        angles = (2.0 * torch.rand(B, generator=gen, dtype=dtype, device=device) - 1.0) * cfg.max_angle_hinge
        max_moved = max(m[4] - m[3] for m in meta)
        n_moved_list = [m[4] - m[3] for m in meta]
        old_batch = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
        pad_mask = torch.zeros(B, max_moved, dtype=torch.bool, device=device)
        for bi, (ci, _, _, ms, me, _) in enumerate(meta):
            nm = me - ms
            old_batch[bi, :nm] = state.positions[ci, ms:me]
            pad_mask[bi, :nm] = True
        delta = old_batch - pos_a[:, None, :]
        delta = delta - box * torch.round(delta * inv_box)
        angles = MoveProposer.cap_angles_batched_by_delta(
            angles, axes, delta, pad_mask, cfg.l0, MoveProposer.BOND_TOL)
        R_batch = MoveProposer._batched_rodrigues(axes, angles, valid)
        rotated = torch.bmm(delta, R_batch.transpose(1, 2))
        new_batch = pos_a[:, None, :] + rotated
        new_batch = ns.wrap(new_batch)
        results = []
        for bi, (ci, _, _, ms, me, mtype) in enumerate(meta):
            nm = n_moved_list[bi]
            old_pos = old_batch[bi, :nm].clone()
            new_pos = new_batch[bi, :nm] if valid[bi] else old_pos.clone()
            results.append(MoveProposal(ci, ms, me, old_pos, new_pos, mtype))
        return results

    @staticmethod
    def propose_batch_segment_moves_fused(state: ChainState, segment_list,
                                          gen, cfg: SimulationConfig):
        B = len(segment_list)
        N = cfg.N
        ns = state.ns
        device = cfg.get_torch_device()
        dtype = cfg.dtype
        box = ns.box_size; inv_box = ns._inv_box
        seg_info = state.segments
        chain_py = [0] * B
        a_idx_py = [0] * B
        b_idx_py = [0] * B
        ms_py = [0] * B
        me_py = [0] * B
        mtype_py = [''] * B
        is_ntail_py = [False] * B
        for bi, (chain_idx, local_seg) in enumerate(segment_list):
            seg_type = seg_info.get_segment_type(local_seg)
            seg_start, seg_end = seg_info.get_segment_range(chain_idx, local_seg)
            if seg_type == SegmentInfo.INNER:
                ai = max(seg_start - 1, 0); bi_ = min(seg_end, N - 1)
                ms, me = seg_start, seg_end; mt = 'hinge'
            elif seg_type in (SegmentInfo.N_TERMINAL, SegmentInfo.BOTH):
                ai = min(seg_end, N - 1); bi_ = min(seg_end + 1, N - 1)
                ms, me = 0, seg_end; mt = 'n_tail'
            else:
                ai = max(seg_start - 1, 0); bi_ = max(seg_start - 2, 0)
                ms, me = seg_start, N; mt = 'c_tail'
            chain_py[bi] = chain_idx; a_idx_py[bi] = ai; b_idx_py[bi] = bi_
            ms_py[bi] = ms; me_py[bi] = me; mtype_py[bi] = mt
            is_ntail_py[bi] = (mt == 'n_tail')
        max_moved = max(me_py[bi] - ms_py[bi] for bi in range(B))
        max_moved = max(max_moved, 1)

        positions_flat = state.get_all_flat()
        chain_t = torch.as_tensor(chain_py, dtype=torch.long, device=device)
        a_idx_t = torch.as_tensor(a_idx_py, dtype=torch.long, device=device)
        b_idx_t = torch.as_tensor(b_idx_py, dtype=torch.long, device=device)
        ms_t = torch.as_tensor(ms_py, dtype=torch.long, device=device)
        me_t = torch.as_tensor(me_py, dtype=torch.long, device=device)
        is_ntail_t = torch.as_tensor(is_ntail_py, dtype=torch.bool, device=device)
        n_moved_t = me_t - ms_t

        a_global = chain_t * N + a_idx_t
        b_global = chain_t * N + b_idx_t
        pos_a = positions_flat[a_global]
        pos_b = positions_flat[b_global]
        axes = pos_b - pos_a
        axes = axes - box * torch.round(axes * inv_box)
        axis_norms = torch.norm(axes, dim=1)
        valid_mask = axis_norms >= MoveProposer.AXIS_EPS

        # Vectorised degen fallback (no .item(), no Python loop).
        # For every row we always compute a fallback axis; we only swap it in
        # where the primary axis was degenerate AND the tangent-cross produced
        # a non-zero axis. All rows pay one extra cross product — cheap vs a
        # sync-per-degen-row.
        c_idx_t = torch.clamp(a_idx_t + 1, max=N - 1)
        c_global = chain_t * N + c_idx_t
        tang = positions_flat[c_global] - positions_flat[a_global]
        tang = tang - box * torch.round(tang * inv_box)
        t_norm = torch.norm(tang, dim=1)
        t_ok = t_norm >= MoveProposer.AXIS_EPS
        safe_tnorm = torch.where(t_ok, t_norm, torch.ones_like(t_norm))
        tang0_ratio = tang[:, 0].abs() / safe_tnorm
        ref_idx_t = (tang0_ratio > 0.9).long()  # 0 or 1 per row
        ref_vec = torch.zeros(B, 3, dtype=dtype, device=device)
        ref_vec.scatter_(1, ref_idx_t.unsqueeze(1), 1.0)
        fb_axis = torch.cross(tang, ref_vec, dim=1)
        fb_norm = torch.norm(fb_axis, dim=1)
        fb_ok = fb_norm >= MoveProposer.AXIS_EPS
        degen = ~valid_mask
        replace = degen & t_ok & fb_ok
        axes = torch.where(replace.unsqueeze(1), fb_axis, axes)
        valid_mask = valid_mask | replace

        angles = (2.0 * torch.rand(B, generator=gen, dtype=dtype, device=device) - 1.0) * cfg.max_angle_hinge

        # Vectorised gather of old positions (natural order) and
        # anchor-relative unwrap. For n_tail rows the input to the unwrap has
        # the beads reversed (anchor-adjacent bead first) so cumsum stays on
        # the correct side of the PBC; we then undo the reversal to land the
        # delta back in natural order.
        k_range = torch.arange(max_moved, device=device, dtype=torch.long)
        valid_lane = k_range.unsqueeze(0) < n_moved_t.unsqueeze(1)  # [B, M]
        nat_idx = (ms_t.unsqueeze(1) + k_range.unsqueeze(0)).clamp(min=0, max=N - 1)
        rev_idx = (me_t.unsqueeze(1) - 1 - k_range.unsqueeze(0)).clamp(min=0, max=N - 1)
        unwrap_idx = torch.where(is_ntail_t.unsqueeze(1), rev_idx, nat_idx)
        chain_exp = chain_t.unsqueeze(1).expand(B, max_moved)
        old_batch = state.positions[chain_exp, nat_idx]  # natural order
        old_batch = old_batch * valid_lane.unsqueeze(-1).to(dtype)
        beads = state.positions[chain_exp, unwrap_idx]
        uw = ns.unwrap_segments_from_anchor(beads, pos_a, valid_lane)
        # For n_tail rows, the k-th bead in natural order corresponds to
        # uw[bi, n_moved-1-k]. Build a gather index and pick.
        nat_to_uw_nt = (n_moved_t - 1).unsqueeze(1) - k_range.unsqueeze(0)
        nat_to_uw_nt = nat_to_uw_nt.clamp(min=0, max=max_moved - 1)
        nat_to_uw = torch.where(is_ntail_t.unsqueeze(1),
                                nat_to_uw_nt, k_range.unsqueeze(0))
        uw_nat = torch.gather(uw, 1, nat_to_uw.unsqueeze(-1).expand(B, max_moved, 3))
        delta = (uw_nat - pos_a.unsqueeze(1)) * valid_lane.unsqueeze(-1).to(dtype)
        pad_mask = valid_lane

        angles = MoveProposer.cap_angles_batched_by_delta(
            angles, axes, delta, pad_mask, cfg.l0, MoveProposer.BOND_TOL)
        R_batch = MoveProposer._batched_rodrigues(axes, angles, valid_mask)
        rotated = torch.bmm(delta, R_batch.transpose(1, 2))
        new_batch = pos_a[:, None, :] + rotated
        new_batch = ns.wrap(new_batch)
        # Invalid rows: restore old. Unconditional masked copy — skipping the
        # `.any()` guard buys one fewer sync per batch.
        invalid_mask = (~valid_mask).view(B, 1, 1)
        new_batch = torch.where(invalid_mask, old_batch, new_batch)
        bp = BatchProposal(B, max_moved, device, dtype)
        bp.old_pos = old_batch
        bp.new_pos = new_batch
        bp.chain_idx = list(chain_py)
        bp.bead_start = list(ms_py)
        bp.n_moved = [me_py[bi] - ms_py[bi] for bi in range(B)]
        bp.move_types = list(mtype_py)
        # bp.valid is metadata only; the sweep gates on n_moved, not valid.
        # Avoid a D2H sync on valid_mask — set a best-effort Python estimate.
        bp.valid = [bp.n_moved[bi] > 0 for bi in range(B)]
        bp.chain_idx_t = chain_t
        bp.bead_start_t = ms_t
        bp.n_moved_t = n_moved_t
        return bp
