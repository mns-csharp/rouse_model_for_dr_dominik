"""Energy module — parallel three-zone delta-E + sparse rank-1 EMM correction.

Highlights vs. single-core variant:
  - `batch_delta_e_par`: prange over the B proposals.
  - `correction_matrix_sparse`: per-pair (i, j) bounding-sphere prefilter
    skips disjoint proposals; only overlapping pairs evaluate the rank-1
    energy. Output is a dense [B, B] array but is mostly zero in practice.
  - `_pair_index` maps a flat pair index k -> (i, j) for prange parallel
    iteration over the upper-triangular pair set.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False)
def _pair_energy(r2, r_rep_sq, r_max_sq, rep_e, contact_e):
    if r2 < r_rep_sq:
        return rep_e
    if contact_e != 0.0 and r2 < r_max_sq:
        return contact_e
    return 0.0


@njit(cache=True, fastmath=False)
def _seq_skip(ci, ri, ti, cj, rj, tj,
              min_caca, min_casg, min_sgsg):
    if ci != cj:
        return False
    seq = ri - rj
    if seq < 0:
        seq = -seq
    if ti == 0 and tj == 0:
        return seq < min_caca
    if ti == 1 and tj == 1:
        return seq < min_sgsg
    return seq < min_casg


@njit(parallel=True, nogil=True, cache=True, fastmath=False)
def build_cell_list_par(flat_pos, nc, inv_cs, half_box):
    n = flat_pos.shape[0]
    nc_m1 = nc - 1
    nc3 = nc * nc * nc
    cell_idx = np.empty(n, dtype=np.int64)
    for f in prange(n):
        cx = int((flat_pos[f, 0] + half_box) * inv_cs)
        cy = int((flat_pos[f, 1] + half_box) * inv_cs)
        cz = int((flat_pos[f, 2] + half_box) * inv_cs)
        if cx < 0: cx = 0
        elif cx > nc_m1: cx = nc_m1
        if cy < 0: cy = 0
        elif cy > nc_m1: cy = nc_m1
        if cz < 0: cz = 0
        elif cz > nc_m1: cz = nc_m1
        cell_idx[f] = (cx * nc + cy) * nc + cz
    sorted_order = np.argsort(cell_idx)
    counts = np.zeros(nc3, dtype=np.int64)
    for f in range(n):
        counts[cell_idx[f]] += 1
    starts = np.empty(nc3, dtype=np.int64)
    starts[0] = 0
    for c in range(1, nc3):
        starts[c] = starts[c - 1] + counts[c - 1]
    return sorted_order, starts, counts


@njit(parallel=True, nogil=True, cache=True, fastmath=False)
def batch_delta_e_par(
        B, old_ca, new_ca, old_sg, new_sg, n_moved,
        chain_idx_arr, bead_start_arr,
        N, flat_pos,
        nc, inv_cs, half_box, neighbor_offsets,
        sorted_order, cell_starts, cell_counts,
        box, inv_box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg):
    out = np.zeros(B, dtype=np.float64)
    nc_m1 = nc - 1
    nc3 = nc * nc * nc
    for b in prange(B):
        nm = n_moved[b]
        if nm == 0:
            continue
        chain_idx = chain_idx_arr[b]
        ms = bead_start_arr[b]
        moved_first = 2 * (chain_idx * N + ms)
        moved_last = 2 * (chain_idx * N + ms + nm) - 1
        visited = np.zeros(nc3, dtype=np.uint8)
        for k in range(nm):
            for src in range(4):
                if src == 0:
                    px, py, pz = old_ca[b, k, 0], old_ca[b, k, 1], old_ca[b, k, 2]
                elif src == 1:
                    px, py, pz = new_ca[b, k, 0], new_ca[b, k, 1], new_ca[b, k, 2]
                elif src == 2:
                    px, py, pz = old_sg[b, k, 0], old_sg[b, k, 1], old_sg[b, k, 2]
                else:
                    px, py, pz = new_sg[b, k, 0], new_sg[b, k, 1], new_sg[b, k, 2]
                cx = int((px + half_box) * inv_cs)
                cy = int((py + half_box) * inv_cs)
                cz = int((pz + half_box) * inv_cs)
                if cx < 0: cx = 0
                elif cx > nc_m1: cx = nc_m1
                if cy < 0: cy = 0
                elif cy > nc_m1: cy = nc_m1
                if cz < 0: cz = 0
                elif cz > nc_m1: cz = nc_m1
                cell = (cx * nc + cy) * nc + cz
                for ni in range(27):
                    visited[neighbor_offsets[cell, ni]] = 1
        e_old = 0.0
        e_new = 0.0
        for c in range(nc3):
            if visited[c] == 0: continue
            cnt = cell_counts[c]
            if cnt == 0: continue
            s = cell_starts[c]
            for ii in range(cnt):
                atom_f = sorted_order[s + ii]
                if atom_f >= moved_first and atom_f <= moved_last:
                    continue
                ax = flat_pos[atom_f, 0]; ay = flat_pos[atom_f, 1]; az = flat_pos[atom_f, 2]
                atom_t = atom_f & 1
                atom_res = atom_f >> 1
                atom_chain = atom_res // N
                atom_ri = atom_res - atom_chain * N
                for k in range(nm):
                    resi = ms + k
                    if not _seq_skip(chain_idx, resi, 0, atom_chain, atom_ri, atom_t,
                                     min_caca, min_casg, min_sgsg):
                        dx = ax - old_ca[b, k, 0]; dy = ay - old_ca[b, k, 1]; dz = az - old_ca[b, k, 2]
                        dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                        e_old += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                        dx = ax - new_ca[b, k, 0]; dy = ay - new_ca[b, k, 1]; dz = az - new_ca[b, k, 2]
                        dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                        e_new += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                    if not _seq_skip(chain_idx, resi, 1, atom_chain, atom_ri, atom_t,
                                     min_caca, min_casg, min_sgsg):
                        dx = ax - old_sg[b, k, 0]; dy = ay - old_sg[b, k, 1]; dz = az - old_sg[b, k, 2]
                        dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                        e_old += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                        dx = ax - new_sg[b, k, 0]; dy = ay - new_sg[b, k, 1]; dz = az - new_sg[b, k, 2]
                        dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                        e_new += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
        out[b] = e_new - e_old
    return out


@njit(cache=True, inline="always")
def _pair_index(k, B):
    a = 2.0 * B - 1.0
    discr = a * a - 8.0 * float(k)
    if discr < 0.0:
        discr = 0.0
    i = int((a - math.sqrt(discr)) * 0.5)
    base_i = i * (2 * B - i - 1) // 2
    while base_i + (B - i - 1) <= k:
        i += 1
        base_i = i * (2 * B - i - 1) // 2
    while base_i > k:
        i -= 1
        base_i = i * (2 * B - i - 1) // 2
    j = i + 1 + (k - base_i)
    return i, j


@njit(parallel=True, nogil=True, cache=True, fastmath=False)
def correction_matrix_sparse(
        B,
        old_ca, new_ca, old_sg, new_sg, n_moved,
        chain_idx_arr, bead_start_arr, N,
        box, inv_box, r_max,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg):
    """Sparse rank-1 correction. For each (i, j) with i<j:
      - If proposals i and j touch disjoint regions (centroid distance
        > radius_i + radius_j + r_max) skip — correction is 0.
      - Otherwise compute the four rank-1 components.
    """
    corr = np.zeros((B, B), dtype=np.float64)
    n_pairs = B * (B - 1) // 2
    if n_pairs <= 0:
        return corr

    # Pre-compute per-proposal centroid (over old+new, ca+sg) and bounding radius.
    cent = np.zeros((B, 3), dtype=np.float64)
    radius = np.zeros(B, dtype=np.float64)
    for b in prange(B):
        nm = n_moved[b]
        if nm == 0:
            continue
        sx = 0.0; sy = 0.0; sz = 0.0; cnt = 0
        for k in range(nm):
            sx += old_ca[b, k, 0]; sy += old_ca[b, k, 1]; sz += old_ca[b, k, 2]
            sx += new_ca[b, k, 0]; sy += new_ca[b, k, 1]; sz += new_ca[b, k, 2]
            sx += old_sg[b, k, 0]; sy += old_sg[b, k, 1]; sz += old_sg[b, k, 2]
            sx += new_sg[b, k, 0]; sy += new_sg[b, k, 1]; sz += new_sg[b, k, 2]
            cnt += 4
        cx = sx / cnt; cy = sy / cnt; cz = sz / cnt
        cent[b, 0] = cx; cent[b, 1] = cy; cent[b, 2] = cz
        rmax = 0.0
        for k in range(nm):
            for src in range(4):
                if src == 0:
                    px = old_ca[b, k, 0]; py = old_ca[b, k, 1]; pz = old_ca[b, k, 2]
                elif src == 1:
                    px = new_ca[b, k, 0]; py = new_ca[b, k, 1]; pz = new_ca[b, k, 2]
                elif src == 2:
                    px = old_sg[b, k, 0]; py = old_sg[b, k, 1]; pz = old_sg[b, k, 2]
                else:
                    px = new_sg[b, k, 0]; py = new_sg[b, k, 1]; pz = new_sg[b, k, 2]
                dx = px - cx; dy = py - cy; dz = pz - cz
                dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                d = math.sqrt(dx*dx + dy*dy + dz*dz)
                if d > rmax:
                    rmax = d
        radius[b] = rmax

    for k in prange(n_pairs):
        i, j = _pair_index(k, B)
        nm_i = n_moved[i]; nm_j = n_moved[j]
        if nm_i == 0 or nm_j == 0:
            continue
        # Sphere prefilter (MIC).
        cdx = cent[j, 0] - cent[i, 0]
        cdy = cent[j, 1] - cent[i, 1]
        cdz = cent[j, 2] - cent[i, 2]
        cdx -= box * round(cdx * inv_box); cdy -= box * round(cdy * inv_box); cdz -= box * round(cdz * inv_box)
        cd = math.sqrt(cdx*cdx + cdy*cdy + cdz*cdz)
        if cd > radius[i] + radius[j] + r_max:
            continue
        ci = chain_idx_arr[i]; ms_i = bead_start_arr[i]
        cj = chain_idx_arr[j]; ms_j = bead_start_arr[j]
        e00 = 0.0; e01 = 0.0; e10 = 0.0; e11 = 0.0
        for ai in range(nm_i):
            resi = ms_i + ai
            for ti in range(2):
                if ti == 0:
                    oix = old_ca[i, ai, 0]; oiy = old_ca[i, ai, 1]; oiz = old_ca[i, ai, 2]
                    nix = new_ca[i, ai, 0]; niy = new_ca[i, ai, 1]; niz = new_ca[i, ai, 2]
                else:
                    oix = old_sg[i, ai, 0]; oiy = old_sg[i, ai, 1]; oiz = old_sg[i, ai, 2]
                    nix = new_sg[i, ai, 0]; niy = new_sg[i, ai, 1]; niz = new_sg[i, ai, 2]
                for bj in range(nm_j):
                    resj = ms_j + bj
                    for tj in range(2):
                        if _seq_skip(ci, resi, ti, cj, resj, tj,
                                     min_caca, min_casg, min_sgsg):
                            continue
                        if tj == 0:
                            ojx = old_ca[j, bj, 0]; ojy = old_ca[j, bj, 1]; ojz = old_ca[j, bj, 2]
                            njx = new_ca[j, bj, 0]; njy = new_ca[j, bj, 1]; njz = new_ca[j, bj, 2]
                        else:
                            ojx = old_sg[j, bj, 0]; ojy = old_sg[j, bj, 1]; ojz = old_sg[j, bj, 2]
                            njx = new_sg[j, bj, 0]; njy = new_sg[j, bj, 1]; njz = new_sg[j, bj, 2]
                        dx = ojx - oix; dy = ojy - oiy; dz = ojz - oiz
                        dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                        e00 += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                        dx = njx - oix; dy = njy - oiy; dz = njz - oiz
                        dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                        e01 += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                        dx = ojx - nix; dy = ojy - niy; dz = ojz - niz
                        dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                        e10 += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                        dx = njx - nix; dy = njy - niy; dz = njz - niz
                        dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                        e11 += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
        delta = (e11 - e01) - (e10 - e00)
        if delta != 0.0:
            corr[i, j] = delta
    return corr
