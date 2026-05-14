"""Move proposer — parallel batch generation across all B proposals.

`@njit(parallel=True)` with `prange` over batch dim. Per-thread Rodrigues
rotations operate on independent batch rows so there's no synchronization
between threads. RNG is pre-drawn on the host and indexed by proposal id
for deterministic results regardless of thread count.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


SEG_N_TERMINAL = 0
SEG_C_TERMINAL = 1
SEG_INNER = 2
SEG_BOTH = 3

MTYPE_HINGE = 0
MTYPE_N_TAIL = 1
MTYPE_C_TAIL = 2

AXIS_EPS = 1e-7
BOND_TOL = 0.05


@njit(parallel=True, nogil=True, cache=True, fastmath=False)
def propose_batch_par(
        ca, sg,
        chain_idx_arr, bead_start_arr, n_moved_arr_in,
        seg_type_arr, anchor_a_local, anchor_b_local,
        randoms,
        old_ca, new_ca, old_sg, new_sg,
        n_moved_arr_out, move_type_arr,
        N, box, inv_box, half_box,
        max_angle, l0,
        cap_inner, cap_tail,
):
    B = chain_idx_arr.shape[0]
    bond_tol_disp = l0 * (1.0 + BOND_TOL)
    for pi in prange(B):
        ci = chain_idx_arr[pi]
        stype = seg_type_arr[pi]
        ms = bead_start_arr[pi]
        nm = n_moved_arr_in[pi]
        a_idx = anchor_a_local[pi]
        b_idx = anchor_b_local[pi]

        for k in range(nm):
            old_ca[pi, k, 0] = ca[ci, ms + k, 0]
            old_ca[pi, k, 1] = ca[ci, ms + k, 1]
            old_ca[pi, k, 2] = ca[ci, ms + k, 2]
            old_sg[pi, k, 0] = sg[ci, ms + k, 0]
            old_sg[pi, k, 1] = sg[ci, ms + k, 1]
            old_sg[pi, k, 2] = sg[ci, ms + k, 2]

        is_inner = (stype == SEG_INNER)
        if is_inner:
            move_type_arr[pi] = MTYPE_HINGE
            cap_enable = cap_inner
        elif stype == SEG_C_TERMINAL:
            move_type_arr[pi] = MTYPE_C_TAIL
            cap_enable = cap_tail
        else:
            move_type_arr[pi] = MTYPE_N_TAIL
            cap_enable = cap_tail

        anc_x = ca[ci, a_idx, 0]
        anc_y = ca[ci, a_idx, 1]
        anc_z = ca[ci, a_idx, 2]

        if a_idx <= ms:
            dx = ca[ci, ms, 0] - anc_x
            dy = ca[ci, ms, 1] - anc_y
            dz = ca[ci, ms, 2] - anc_z
            dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
            new_ca[pi, 0, 0] = anc_x + dx
            new_ca[pi, 0, 1] = anc_y + dy
            new_ca[pi, 0, 2] = anc_z + dz
            for k in range(1, nm):
                gx = ca[ci, ms + k, 0] - ca[ci, ms + k - 1, 0]
                gy = ca[ci, ms + k, 1] - ca[ci, ms + k - 1, 1]
                gz = ca[ci, ms + k, 2] - ca[ci, ms + k - 1, 2]
                gx -= box * round(gx * inv_box); gy -= box * round(gy * inv_box); gz -= box * round(gz * inv_box)
                new_ca[pi, k, 0] = new_ca[pi, k - 1, 0] + gx
                new_ca[pi, k, 1] = new_ca[pi, k - 1, 1] + gy
                new_ca[pi, k, 2] = new_ca[pi, k - 1, 2] + gz
        else:
            last = nm - 1
            dx = ca[ci, ms + last, 0] - anc_x
            dy = ca[ci, ms + last, 1] - anc_y
            dz = ca[ci, ms + last, 2] - anc_z
            dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
            new_ca[pi, last, 0] = anc_x + dx
            new_ca[pi, last, 1] = anc_y + dy
            new_ca[pi, last, 2] = anc_z + dz
            for k in range(last - 1, -1, -1):
                gx = ca[ci, ms + k, 0] - ca[ci, ms + k + 1, 0]
                gy = ca[ci, ms + k, 1] - ca[ci, ms + k + 1, 1]
                gz = ca[ci, ms + k, 2] - ca[ci, ms + k + 1, 2]
                gx -= box * round(gx * inv_box); gy -= box * round(gy * inv_box); gz -= box * round(gz * inv_box)
                new_ca[pi, k, 0] = new_ca[pi, k + 1, 0] + gx
                new_ca[pi, k, 1] = new_ca[pi, k + 1, 1] + gy
                new_ca[pi, k, 2] = new_ca[pi, k + 1, 2] + gz

        for k in range(nm):
            ox = sg[ci, ms + k, 0] - ca[ci, ms + k, 0]
            oy = sg[ci, ms + k, 1] - ca[ci, ms + k, 1]
            oz = sg[ci, ms + k, 2] - ca[ci, ms + k, 2]
            ox -= box * round(ox * inv_box); oy -= box * round(oy * inv_box); oz -= box * round(oz * inv_box)
            new_sg[pi, k, 0] = new_ca[pi, k, 0] + ox
            new_sg[pi, k, 1] = new_ca[pi, k, 1] + oy
            new_sg[pi, k, 2] = new_ca[pi, k, 2] + oz

        if is_inner:
            ux_b = ca[ci, b_idx, 0] - ca[ci, ms + nm - 1, 0]
            uy_b = ca[ci, b_idx, 1] - ca[ci, ms + nm - 1, 1]
            uz_b = ca[ci, b_idx, 2] - ca[ci, ms + nm - 1, 2]
            ux_b -= box * round(ux_b * inv_box); uy_b -= box * round(uy_b * inv_box); uz_b -= box * round(uz_b * inv_box)
            ub_x = new_ca[pi, nm - 1, 0] + ux_b
            ub_y = new_ca[pi, nm - 1, 1] + uy_b
            ub_z = new_ca[pi, nm - 1, 2] + uz_b
            ax_dx = ub_x - anc_x
            ax_dy = ub_y - anc_y
            ax_dz = ub_z - anc_z
            axis_len = math.sqrt(ax_dx * ax_dx + ax_dy * ax_dy + ax_dz * ax_dz)
        else:
            ax_dx = ca[ci, b_idx, 0] - anc_x
            ax_dy = ca[ci, b_idx, 1] - anc_y
            ax_dz = ca[ci, b_idx, 2] - anc_z
            ax_dx -= box * round(ax_dx * inv_box); ax_dy -= box * round(ax_dy * inv_box); ax_dz -= box * round(ax_dz * inv_box)
            axis_len = math.sqrt(ax_dx * ax_dx + ax_dy * ax_dy + ax_dz * ax_dz)
            if axis_len < AXIS_EPS:
                c_idx = a_idx + 1
                if c_idx >= N:
                    c_idx = N - 1
                tx = ca[ci, c_idx, 0] - anc_x
                ty = ca[ci, c_idx, 1] - anc_y
                tz = ca[ci, c_idx, 2] - anc_z
                tx -= box * round(tx * inv_box); ty -= box * round(ty * inv_box); tz -= box * round(tz * inv_box)
                t_norm = math.sqrt(tx * tx + ty * ty + tz * tz)
                if t_norm > AXIS_EPS:
                    if abs(tx) / t_norm > 0.9:
                        rx, ry, rz = 0.0, 1.0, 0.0
                    else:
                        rx, ry, rz = 1.0, 0.0, 0.0
                    ax_dx = ty * rz - tz * ry
                    ax_dy = tz * rx - tx * rz
                    ax_dz = tx * ry - ty * rx
                    axis_len = math.sqrt(ax_dx * ax_dx + ax_dy * ax_dy + ax_dz * ax_dz)

        if axis_len < AXIS_EPS:
            n_moved_arr_out[pi] = 0
            for k in range(nm):
                new_ca[pi, k, 0] = old_ca[pi, k, 0]
                new_ca[pi, k, 1] = old_ca[pi, k, 1]
                new_ca[pi, k, 2] = old_ca[pi, k, 2]
                new_sg[pi, k, 0] = old_sg[pi, k, 0]
                new_sg[pi, k, 1] = old_sg[pi, k, 1]
                new_sg[pi, k, 2] = old_sg[pi, k, 2]
            continue

        inv_len = 1.0 / axis_len
        ux = ax_dx * inv_len
        uy = ax_dy * inv_len
        uz = ax_dz * inv_len
        angle = (2.0 * randoms[pi] - 1.0) * max_angle

        if cap_enable:
            max_perp = 0.0
            for k in range(nm):
                rx = new_ca[pi, k, 0] - anc_x
                ry = new_ca[pi, k, 1] - anc_y
                rz = new_ca[pi, k, 2] - anc_z
                proj = rx * ux + ry * uy + rz * uz
                px = rx - proj * ux
                py = ry - proj * uy
                pz = rz - proj * uz
                pn = math.sqrt(px * px + py * py + pz * pz)
                if pn > max_perp:
                    max_perp = pn
            if max_perp > 1e-12:
                s_required = bond_tol_disp / (2.0 * max_perp)
                if s_required < 1.0:
                    s_actual = abs(math.sin(0.5 * angle))
                    if s_actual > s_required:
                        theta = 2.0 * math.asin(s_required)
                        if angle < 0.0:
                            angle = -theta
                        else:
                            angle = theta

        c = math.cos(angle); s = math.sin(angle); t = 1.0 - c
        r00 = t * ux * ux + c
        r01 = t * ux * uy - s * uz
        r02 = t * ux * uz + s * uy
        r10 = t * ux * uy + s * uz
        r11 = t * uy * uy + c
        r12 = t * uy * uz - s * ux
        r20 = t * ux * uz - s * uy
        r21 = t * uy * uz + s * ux
        r22 = t * uz * uz + c

        for k in range(nm):
            rx = new_ca[pi, k, 0] - anc_x
            ry = new_ca[pi, k, 1] - anc_y
            rz = new_ca[pi, k, 2] - anc_z
            nx = anc_x + r00 * rx + r01 * ry + r02 * rz
            ny = anc_y + r10 * rx + r11 * ry + r12 * rz
            nz = anc_z + r20 * rx + r21 * ry + r22 * rz
            nx -= box * math.floor((nx + half_box) * inv_box)
            ny -= box * math.floor((ny + half_box) * inv_box)
            nz -= box * math.floor((nz + half_box) * inv_box)
            new_ca[pi, k, 0] = nx
            new_ca[pi, k, 1] = ny
            new_ca[pi, k, 2] = nz
            sx = new_sg[pi, k, 0] - anc_x
            sy = new_sg[pi, k, 1] - anc_y
            sz = new_sg[pi, k, 2] - anc_z
            mx = anc_x + r00 * sx + r01 * sy + r02 * sz
            my = anc_y + r10 * sx + r11 * sy + r12 * sz
            mz = anc_z + r20 * sx + r21 * sy + r22 * sz
            mx -= box * math.floor((mx + half_box) * inv_box)
            my -= box * math.floor((my + half_box) * inv_box)
            mz -= box * math.floor((mz + half_box) * inv_box)
            new_sg[pi, k, 0] = mx
            new_sg[pi, k, 1] = my
            new_sg[pi, k, 2] = mz

        n_moved_arr_out[pi] = nm
