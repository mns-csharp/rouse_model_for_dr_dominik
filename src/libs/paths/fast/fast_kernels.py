"""FastKernels — Numba-JIT'd core kernels for the numpy/numba fast backend.

Primary API is the FastKernels class; all @njit functions are exposed as
@staticmethod attributes. The functions remain defined at module scope
(with a leading underscore) so they can resolve one another by name at
JIT-compile time — Numba cannot do class-attribute lookup inside a
compiled function.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _count_overlaps(moved_pos, nbr_pos, box, inv_box, r_rep_sq):
    M = moved_pos.shape[0]
    K = nbr_pos.shape[0]
    count = 0
    for i in range(M):
        mx = moved_pos[i, 0]
        my = moved_pos[i, 1]
        mz = moved_pos[i, 2]
        for j in range(K):
            dx = nbr_pos[j, 0] - mx
            dy = nbr_pos[j, 1] - my
            dz = nbr_pos[j, 2] - mz
            dx -= box * round(dx * inv_box)
            dy -= box * round(dy * inv_box)
            dz -= box * round(dz * inv_box)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < r_rep_sq:
                count += 1
    return count


@njit(cache=True)
def _three_zone_energy(moved_pos, nbr_pos, box, inv_box,
                       r_rep_sq, r_max_sq, rep_e, contact_e):
    M = moved_pos.shape[0]
    K = nbr_pos.shape[0]
    energy = 0.0
    for i in range(M):
        mx = moved_pos[i, 0]
        my = moved_pos[i, 1]
        mz = moved_pos[i, 2]
        for j in range(K):
            dx = nbr_pos[j, 0] - mx
            dy = nbr_pos[j, 1] - my
            dz = nbr_pos[j, 2] - mz
            dx -= box * round(dx * inv_box)
            dy -= box * round(dy * inv_box)
            dz -= box * round(dz * inv_box)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < r_rep_sq:
                energy += rep_e
            elif r2 < r_max_sq:
                energy += contact_e
    return energy


@njit(cache=True)
def _delta_energy_fast(old_pos, new_pos, nbr_pos, box, inv_box, r_rep_sq, rep_e):
    e_old = _count_overlaps(old_pos, nbr_pos, box, inv_box, r_rep_sq)
    e_new = _count_overlaps(new_pos, nbr_pos, box, inv_box, r_rep_sq)
    return (e_new - e_old) * rep_e


@njit(cache=True)
def _delta_energy_3zone(old_pos, new_pos, nbr_pos, box, inv_box,
                        r_rep_sq, r_max_sq, rep_e, contact_e):
    e_old = _three_zone_energy(old_pos, nbr_pos, box, inv_box,
                               r_rep_sq, r_max_sq, rep_e, contact_e)
    e_new = _three_zone_energy(new_pos, nbr_pos, box, inv_box,
                               r_rep_sq, r_max_sq, rep_e, contact_e)
    return e_new - e_old


@njit(cache=True)
def _compute_segment_pair_energy_fast(pos_a, pos_b, box, inv_box, r_rep_sq, rep_e):
    return _count_overlaps(pos_a, pos_b, box, inv_box, r_rep_sq) * rep_e


@njit(cache=True)
def _gather_neighbors_jit(old_pos, new_pos, nc, inv_cs, half_box,
                          neighbor_offsets, sorted_order, cell_starts,
                          cell_counts, global_exclude_start, global_exclude_end):
    nc_m1 = nc - 1
    M_old = old_pos.shape[0]
    M_new = new_pos.shape[0]
    nc3 = nc * nc * nc

    visited = np.zeros(nc3, dtype=np.bool_)

    for k in range(M_old):
        cx = max(0, min(nc_m1, int((old_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((old_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((old_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            visited[neighbor_offsets[cell, ni]] = True

    for k in range(M_new):
        cx = max(0, min(nc_m1, int((new_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((new_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((new_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            visited[neighbor_offsets[cell, ni]] = True

    total = 0
    for c in range(nc3):
        if visited[c]:
            total += cell_counts[c]

    result = np.empty(total, dtype=np.int64)
    idx = 0
    gs = global_exclude_start
    ge = global_exclude_end
    for c in range(nc3):
        if not visited[c]:
            continue
        cnt = cell_counts[c]
        if cnt == 0:
            continue
        s = cell_starts[c]
        for i in range(cnt):
            atom = sorted_order[s + i]
            if atom < gs or atom >= ge:
                result[idx] = atom
                idx += 1

    return result[:idx]


@njit(cache=True)
def _delta_energy_direct(old_pos, new_pos, all_pos, nc, inv_cs, half_box,
                         neighbor_offsets, sorted_order, cell_starts,
                         cell_counts, global_exclude_start, global_exclude_end,
                         box, inv_box, r_rep_sq, rep_e):
    nc_m1 = nc - 1
    M_old = old_pos.shape[0]
    M_new = new_pos.shape[0]
    nc3 = nc * nc * nc
    gs = global_exclude_start
    ge = global_exclude_end

    visited = np.zeros(nc3, dtype=np.bool_)
    visited_list = np.empty((M_old + M_new) * 27, dtype=np.int64)
    n_visited = 0

    for k in range(M_old):
        cx = max(0, min(nc_m1, int((old_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((old_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((old_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            c = neighbor_offsets[cell, ni]
            if not visited[c]:
                visited[c] = True
                visited_list[n_visited] = c
                n_visited += 1
    for k in range(M_new):
        cx = max(0, min(nc_m1, int((new_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((new_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((new_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            c = neighbor_offsets[cell, ni]
            if not visited[c]:
                visited[c] = True
                visited_list[n_visited] = c
                n_visited += 1

    e_old = 0
    e_new = 0
    for vi in range(n_visited):
        c = visited_list[vi]
        cnt = cell_counts[c]
        if cnt == 0:
            continue
        s = cell_starts[c]
        for i in range(cnt):
            atom = sorted_order[s + i]
            if atom >= gs and atom < ge:
                continue
            nx = all_pos[atom, 0]
            ny = all_pos[atom, 1]
            nz = all_pos[atom, 2]
            for m in range(M_old):
                dx = nx - old_pos[m, 0]
                dy = ny - old_pos[m, 1]
                dz = nz - old_pos[m, 2]
                dx -= box * round(dx * inv_box)
                dy -= box * round(dy * inv_box)
                dz -= box * round(dz * inv_box)
                r2 = dx*dx + dy*dy + dz*dz
                if r2 < r_rep_sq:
                    e_old += 1
            for m in range(M_new):
                dx = nx - new_pos[m, 0]
                dy = ny - new_pos[m, 1]
                dz = nz - new_pos[m, 2]
                dx -= box * round(dx * inv_box)
                dy -= box * round(dy * inv_box)
                dz -= box * round(dz * inv_box)
                r2 = dx*dx + dy*dy + dz*dz
                if r2 < r_rep_sq:
                    e_new += 1

    return (e_new - e_old) * rep_e


@njit(cache=True)
def _delta_energy_direct_3zone(old_pos, new_pos, all_pos, nc, inv_cs, half_box,
                               neighbor_offsets, sorted_order, cell_starts,
                               cell_counts, global_exclude_start, global_exclude_end,
                               box, inv_box, r_rep_sq, r_max_sq, rep_e, contact_e):
    nc_m1 = nc - 1
    M_old = old_pos.shape[0]
    M_new = new_pos.shape[0]
    nc3 = nc * nc * nc
    gs = global_exclude_start
    ge = global_exclude_end

    visited = np.zeros(nc3, dtype=np.bool_)
    visited_list = np.empty((M_old + M_new) * 27, dtype=np.int64)
    n_visited = 0

    for k in range(M_old):
        cx = max(0, min(nc_m1, int((old_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((old_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((old_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            c = neighbor_offsets[cell, ni]
            if not visited[c]:
                visited[c] = True
                visited_list[n_visited] = c
                n_visited += 1
    for k in range(M_new):
        cx = max(0, min(nc_m1, int((new_pos[k, 0] + half_box) * inv_cs)))
        cy = max(0, min(nc_m1, int((new_pos[k, 1] + half_box) * inv_cs)))
        cz = max(0, min(nc_m1, int((new_pos[k, 2] + half_box) * inv_cs)))
        cell = (cx * nc + cy) * nc + cz
        for ni in range(27):
            c = neighbor_offsets[cell, ni]
            if not visited[c]:
                visited[c] = True
                visited_list[n_visited] = c
                n_visited += 1

    e_old = 0.0
    e_new = 0.0
    for vi in range(n_visited):
        c = visited_list[vi]
        cnt = cell_counts[c]
        if cnt == 0:
            continue
        s = cell_starts[c]
        for i in range(cnt):
            atom = sorted_order[s + i]
            if atom >= gs and atom < ge:
                continue
            nx = all_pos[atom, 0]
            ny = all_pos[atom, 1]
            nz = all_pos[atom, 2]
            for m in range(M_old):
                dx = nx - old_pos[m, 0]
                dy = ny - old_pos[m, 1]
                dz = nz - old_pos[m, 2]
                dx -= box * round(dx * inv_box)
                dy -= box * round(dy * inv_box)
                dz -= box * round(dz * inv_box)
                r2 = dx*dx + dy*dy + dz*dz
                if r2 < r_rep_sq:
                    e_old += rep_e
                elif r2 < r_max_sq:
                    e_old += contact_e
            for m in range(M_new):
                dx = nx - new_pos[m, 0]
                dy = ny - new_pos[m, 1]
                dz = nz - new_pos[m, 2]
                dx -= box * round(dx * inv_box)
                dy -= box * round(dy * inv_box)
                dz -= box * round(dz * inv_box)
                r2 = dx*dx + dy*dy + dz*dz
                if r2 < r_rep_sq:
                    e_new += rep_e
                elif r2 < r_max_sq:
                    e_new += contact_e

    return e_new - e_old


@njit(parallel=True, cache=True)
def _batch_delta_energy_direct(
        n_proposals,
        old_pos_flat, new_pos_flat, n_moved_arr,
        all_pos, global_starts, global_ends,
        nc, inv_cs, half_box,
        neighbor_offsets, sorted_order, cell_starts, cell_counts,
        box, inv_box, r_rep_sq, rep_e,
        max_moved):
    B = n_proposals
    nc_m1 = nc - 1
    nc3 = nc * nc * nc
    results = np.zeros(B, dtype=np.float64)

    for pi in prange(B):
        nm = n_moved_arr[pi]
        if nm == 0:
            continue
        gs = global_starts[pi]
        ge = global_ends[pi]

        old_p = old_pos_flat[pi, :nm, :]
        new_p = new_pos_flat[pi, :nm, :]

        visited = np.zeros(nc3, dtype=np.bool_)
        for k in range(nm):
            cx = max(0, min(nc_m1, int((old_p[k, 0] + half_box) * inv_cs)))
            cy = max(0, min(nc_m1, int((old_p[k, 1] + half_box) * inv_cs)))
            cz = max(0, min(nc_m1, int((old_p[k, 2] + half_box) * inv_cs)))
            cell = (cx * nc + cy) * nc + cz
            for ni in range(27):
                visited[neighbor_offsets[cell, ni]] = True
            cx = max(0, min(nc_m1, int((new_p[k, 0] + half_box) * inv_cs)))
            cy = max(0, min(nc_m1, int((new_p[k, 1] + half_box) * inv_cs)))
            cz = max(0, min(nc_m1, int((new_p[k, 2] + half_box) * inv_cs)))
            cell = (cx * nc + cy) * nc + cz
            for ni in range(27):
                visited[neighbor_offsets[cell, ni]] = True

        e_old = 0
        e_new = 0
        for c in range(nc3):
            if not visited[c]:
                continue
            cnt = cell_counts[c]
            if cnt == 0:
                continue
            s = cell_starts[c]
            for i in range(cnt):
                atom = sorted_order[s + i]
                if atom >= gs and atom < ge:
                    continue
                nx = all_pos[atom, 0]
                ny = all_pos[atom, 1]
                nz = all_pos[atom, 2]
                for m in range(nm):
                    dx = nx - old_p[m, 0]
                    dy = ny - old_p[m, 1]
                    dz = nz - old_p[m, 2]
                    dx -= box * round(dx * inv_box)
                    dy -= box * round(dy * inv_box)
                    dz -= box * round(dz * inv_box)
                    r2 = dx*dx + dy*dy + dz*dz
                    if r2 < r_rep_sq:
                        e_old += 1
                for m in range(nm):
                    dx = nx - new_p[m, 0]
                    dy = ny - new_p[m, 1]
                    dz = nz - new_p[m, 2]
                    dx -= box * round(dx * inv_box)
                    dy -= box * round(dy * inv_box)
                    dz -= box * round(dz * inv_box)
                    r2 = dx*dx + dy*dy + dz*dz
                    if r2 < r_rep_sq:
                        e_new += 1

        results[pi] = (e_new - e_old) * rep_e

    return results


class FastKernels:
    count_overlaps = staticmethod(_count_overlaps)
    three_zone_energy = staticmethod(_three_zone_energy)
    delta_energy_fast = staticmethod(_delta_energy_fast)
    delta_energy_3zone = staticmethod(_delta_energy_3zone)
    compute_segment_pair_energy = staticmethod(_compute_segment_pair_energy_fast)
    gather_neighbors = staticmethod(_gather_neighbors_jit)
    delta_energy_direct = staticmethod(_delta_energy_direct)
    delta_energy_direct_3zone = staticmethod(_delta_energy_direct_3zone)
    batch_delta_energy_direct = staticmethod(_batch_delta_energy_direct)
