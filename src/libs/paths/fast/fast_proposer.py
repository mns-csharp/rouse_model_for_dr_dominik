"""FastProposer — proposal generation (hinge/tail/pivot) for the numpy/numba backend.

All proposal builders are @staticmethod so they can be called without state.
Helpers (Rodrigues matrix, MIC displacement, wrap, unwrap-from-anchor, rotate-and-wrap)
are also @staticmethod on the same class to keep them namespaced.
"""

import math

import numpy as np

from rouse_model_python.src.libs.chain.segment_info import SegmentInfo
from rouse_model_python.src.libs.paths.fast.fast_proposal import FastProposal


class FastProposer:
    AXIS_EPS = 1e-7
    BOND_TOL = 0.05

    @staticmethod
    def rodrigues_matrix(ux: float, uy: float, uz: float, angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        return np.array([
            [t*ux*ux + c,      t*ux*uy - s*uz,  t*ux*uz + s*uy],
            [t*ux*uy + s*uz,   t*uy*uy + c,     t*uy*uz - s*ux],
            [t*ux*uz - s*uy,   t*uy*uz + s*ux,  t*uz*uz + c   ],
        ])

    @staticmethod
    def random_so3_axis_angle(rng):
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
        return ux, uy, uz, angle

    @staticmethod
    def random_so3_matrix(rng) -> np.ndarray:
        ux, uy, uz, angle = FastProposer.random_so3_axis_angle(rng)
        return FastProposer.rodrigues_matrix(ux, uy, uz, angle)

    @staticmethod
    def max_perp_to_axis(unwrapped: np.ndarray, anchor: np.ndarray,
                         ux: float, uy: float, uz: float) -> float:
        rel = unwrapped - anchor[np.newaxis, :]
        proj = rel[:, 0] * ux + rel[:, 1] * uy + rel[:, 2] * uz
        perp = rel - np.outer(proj, np.array([ux, uy, uz]))
        return float(np.linalg.norm(perp, axis=1).max())

    @staticmethod
    def cap_angle_by_displacement(angle: float, max_perp: float,
                                  l0: float, tol: float = 0.05) -> float:
        # Displacement of a bead at perpendicular distance r_perp after a
        # rotation by angle around the axis equals 2 * r_perp * |sin(angle/2)|.
        # Cap |angle| so that max_perp * 2 * |sin(angle/2)| <= l0 * (1 + tol).
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
    def mic_delta_3(ax, ay, az, bx, by, bz, box: float, inv_box: float):
        dx = bx - ax; dy = by - ay; dz = bz - az
        dx -= box * round(dx * inv_box)
        dy -= box * round(dy * inv_box)
        dz -= box * round(dz * inv_box)
        return dx, dy, dz

    @staticmethod
    def wrap_pos(pos: np.ndarray, box: float, half_box: float, inv_box: float) -> np.ndarray:
        return pos - box * np.floor((pos + half_box) * inv_box)

    @staticmethod
    def validate_boundary_bonds(positions_np: np.ndarray, chain_idx: int,
                                bead_start: int, n_moved: int,
                                N: int, l0: float, box: float, inv_box: float) -> None:
        tol_sq = (l0 * (1.0 + FastProposer.BOND_TOL)) ** 2
        pos = positions_np[chain_idx]

        if bead_start > 0:
            dx, dy, dz = FastProposer.mic_delta_3(
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
            dx, dy, dz = FastProposer.mic_delta_3(
                pos[bead_end - 1, 0], pos[bead_end - 1, 1], pos[bead_end - 1, 2],
                pos[bead_end, 0], pos[bead_end, 1], pos[bead_end, 2],
                box, inv_box)
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > tol_sq:
                raise RuntimeError(
                    f"Bond broken: chain {chain_idx}, beads "
                    f"{bead_end - 1}-{bead_end}, "
                    f"dist={d2**0.5:.4f} A (l0={l0:.1f} A)")

    @staticmethod
    def unwrap_from_anchor(positions_np: np.ndarray, anchor_idx: int,
                           bead_start: int, bead_end: int,
                           box: float, inv_box: float) -> np.ndarray:
        M = bead_end - bead_start
        unwrapped = np.empty((M, 3))

        if anchor_idx <= bead_start:
            dx, dy, dz = FastProposer.mic_delta_3(
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
            last = M - 1
            gi_last = bead_end - 1
            dx, dy, dz = FastProposer.mic_delta_3(
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

    @staticmethod
    def rotate_and_wrap(unwrapped: np.ndarray, anchor_pos: np.ndarray,
                        R: np.ndarray, box: float, half_box: float,
                        inv_box: float) -> np.ndarray:
        relative = unwrapped - anchor_pos[np.newaxis, :]
        rotated = relative @ R.T
        new_pos = anchor_pos[np.newaxis, :] + rotated
        return FastProposer.wrap_pos(new_pos, box, half_box, inv_box)

    @staticmethod
    def propose_hinge(positions_np: np.ndarray, chain_idx: int,
                      seg_start: int, seg_end: int, N: int,
                      box: float, inv_box: float, half_box: float,
                      max_angle: float, rng, l0: float) -> FastProposal:
        a_idx = max(seg_start - 1, 0)
        b_idx = min(seg_end, N - 1)
        pos_a = positions_np[a_idx]

        old_pos = positions_np[seg_start:seg_end].copy()

        unwrapped = FastProposer.unwrap_from_anchor(
            positions_np, a_idx, seg_start, seg_end, box, inv_box)

        dx = positions_np[b_idx, 0] - positions_np[seg_end - 1, 0]
        dy = positions_np[b_idx, 1] - positions_np[seg_end - 1, 1]
        dz = positions_np[b_idx, 2] - positions_np[seg_end - 1, 2]
        dx -= box * round(dx * inv_box)
        dy -= box * round(dy * inv_box)
        dz -= box * round(dz * inv_box)
        unwrapped_b = [unwrapped[-1, 0] + dx, unwrapped[-1, 1] + dy, unwrapped[-1, 2] + dz]

        ax_dx = unwrapped_b[0] - pos_a[0]
        ax_dy = unwrapped_b[1] - pos_a[1]
        ax_dz = unwrapped_b[2] - pos_a[2]
        axis_len = math.sqrt(ax_dx*ax_dx + ax_dy*ax_dy + ax_dz*ax_dz)

        if axis_len < FastProposer.AXIS_EPS:
            return FastProposal(chain_idx, seg_start, seg_end - seg_start,
                                old_pos, old_pos.copy(), 'hinge')

        inv_len = 1.0 / axis_len
        ux, uy, uz = ax_dx * inv_len, ax_dy * inv_len, ax_dz * inv_len
        angle = (2.0 * rng.random() - 1.0) * max_angle
        max_perp = FastProposer.max_perp_to_axis(unwrapped, pos_a, ux, uy, uz)
        angle = FastProposer.cap_angle_by_displacement(
            angle, max_perp, l0, FastProposer.BOND_TOL)
        R = FastProposer.rodrigues_matrix(ux, uy, uz, angle)

        new_pos = FastProposer.rotate_and_wrap(unwrapped, pos_a, R, box, half_box, inv_box)

        return FastProposal(chain_idx, seg_start, seg_end - seg_start,
                            old_pos, new_pos, 'hinge')

    @staticmethod
    def propose_tail(positions_np: np.ndarray, chain_idx: int,
                     seg_start: int, seg_end: int, is_n_terminal: bool,
                     N: int, box: float, inv_box: float, half_box: float,
                     max_angle: float, rng, l0: float) -> FastProposal:
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

        dx, dy, dz = FastProposer.mic_delta_3(
            pos_a[0], pos_a[1], pos_a[2],
            pos_b[0], pos_b[1], pos_b[2], box, inv_box)
        axis_len = math.sqrt(dx*dx + dy*dy + dz*dz)

        if axis_len < FastProposer.AXIS_EPS:
            c_idx = min(a_idx + 1, N - 1)
            tx, ty, tz = FastProposer.mic_delta_3(
                positions_np[a_idx][0], positions_np[a_idx][1], positions_np[a_idx][2],
                positions_np[c_idx][0], positions_np[c_idx][1], positions_np[c_idx][2],
                box, inv_box)
            t_norm = math.sqrt(tx*tx + ty*ty + tz*tz)
            if t_norm > FastProposer.AXIS_EPS:
                if abs(tx) / t_norm > 0.9:
                    rx, ry, rz = 0.0, 1.0, 0.0
                else:
                    rx, ry, rz = 1.0, 0.0, 0.0
                dx = ty * rz - tz * ry
                dy = tz * rx - tx * rz
                dz = tx * ry - ty * rx
                axis_len = math.sqrt(dx*dx + dy*dy + dz*dz)

            if axis_len < FastProposer.AXIS_EPS:
                old_pos = positions_np[move_start:move_end].copy()
                mtype = 'n_tail' if is_n_terminal else 'c_tail'
                return FastProposal(chain_idx, move_start, move_end - move_start,
                                    old_pos, old_pos.copy(), mtype)

        inv_len = 1.0 / axis_len
        ux, uy, uz = dx * inv_len, dy * inv_len, dz * inv_len
        angle = (2.0 * rng.random() - 1.0) * max_angle

        old_pos = positions_np[move_start:move_end].copy()

        unwrapped = FastProposer.unwrap_from_anchor(
            positions_np, a_idx, move_start, move_end, box, inv_box)
        max_perp = FastProposer.max_perp_to_axis(unwrapped, pos_a, ux, uy, uz)
        angle = FastProposer.cap_angle_by_displacement(
            angle, max_perp, l0, FastProposer.BOND_TOL)
        R = FastProposer.rodrigues_matrix(ux, uy, uz, angle)

        new_pos = FastProposer.rotate_and_wrap(unwrapped, pos_a, R, box, half_box, inv_box)

        mtype = 'n_tail' if is_n_terminal else 'c_tail'
        return FastProposal(chain_idx, move_start, move_end - move_start,
                            old_pos, new_pos, mtype)

    @staticmethod
    def propose_segment_move(positions_np: np.ndarray, chain_idx: int,
                             local_seg: int, seg_info: SegmentInfo,
                             N: int, box: float, inv_box: float, half_box: float,
                             max_angle: float, rng, l0: float) -> FastProposal:
        seg_type = seg_info.get_segment_type(local_seg)
        seg_start, seg_end = seg_info.get_segment_range(chain_idx, local_seg)

        if seg_type == SegmentInfo.INNER:
            return FastProposer.propose_hinge(
                positions_np, chain_idx, seg_start, seg_end,
                N, box, inv_box, half_box, max_angle, rng, l0)
        elif seg_type == SegmentInfo.N_TERMINAL or seg_type == SegmentInfo.BOTH:
            return FastProposer.propose_tail(
                positions_np, chain_idx, seg_start, seg_end,
                True, N, box, inv_box, half_box, max_angle, rng, l0)
        elif seg_type == SegmentInfo.C_TERMINAL:
            return FastProposer.propose_tail(
                positions_np, chain_idx, seg_start, seg_end,
                False, N, box, inv_box, half_box, max_angle, rng, l0)
        else:
            raise ValueError(f"Unknown segment type: {seg_type}")

    @staticmethod
    def propose_pivot(positions_np: np.ndarray, chain_idx: int, N: int,
                      box: float, inv_box: float, half_box: float, rng,
                      l0: float) -> FastProposal:
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

        if side == 0:
            beads = positions_np[rot_start:rot_end][::-1]
        else:
            beads = positions_np[rot_start:rot_end]

        unwrapped = np.empty((n_rot, 3))
        dx, dy, dz = FastProposer.mic_delta_3(
            anchor[0], anchor[1], anchor[2],
            beads[0, 0], beads[0, 1], beads[0, 2], box, inv_box)
        unwrapped[0] = [anchor[0] + dx, anchor[1] + dy, anchor[2] + dz]
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

        ux, uy, uz, angle = FastProposer.random_so3_axis_angle(rng)
        max_perp = FastProposer.max_perp_to_axis(unwrapped, anchor, ux, uy, uz)
        angle = FastProposer.cap_angle_by_displacement(
            angle, max_perp, l0, FastProposer.BOND_TOL)
        R = FastProposer.rodrigues_matrix(ux, uy, uz, angle)
        relative = unwrapped - anchor[np.newaxis, :]
        rotated = relative @ R.T
        new_unwrapped = anchor[np.newaxis, :] + rotated
        new_pos = FastProposer.wrap_pos(new_unwrapped, box, half_box, inv_box)

        if side == 0:
            new_pos = new_pos[::-1].copy()

        old_pos = positions_np[rot_start:rot_end].copy()

        return FastProposal(chain_idx, rot_start, n_rot,
                            old_pos, new_pos, 'pivot')

    @staticmethod
    def metropolis_accept(delta_e: float, kBT: float, rng) -> bool:
        if delta_e <= 0.0:
            return True
        exponent = -delta_e / kBT
        if exponent <= -745.0:
            return False
        if exponent >= 709.0:
            return True
        return rng.random() < math.exp(exponent)
