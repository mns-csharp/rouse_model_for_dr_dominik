"""Energy module — three-zone potential with intra-kernel multicore reduction.

Per-proposal layout: visited cells (3x3x3 around every moved-bead old/new
position) feed a parallel `prange` over the visited-cell list. Each thread
accumulates partial e_old/e_new sums into per-cell entries; final delta_e
is the reduction across cells.
"""

from __future__ import annotations

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
def proposal_delta_e_par(
        old_ca, new_ca, old_sg, new_sg, nm,
        chain_idx, ms, N,
        flat_pos,
        nc, inv_cs, half_box, neighbor_offsets,
        sorted_order, cell_starts, cell_counts,
        box, inv_box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg):
    if nm == 0:
        return 0.0
    nc_m1 = nc - 1
    nc3 = nc * nc * nc

    # Build visited-cell list (sequential — small).
    visited = np.zeros(nc3, dtype=np.uint8)
    for k in range(nm):
        for src in range(4):
            if src == 0:
                px, py, pz = old_ca[k, 0], old_ca[k, 1], old_ca[k, 2]
            elif src == 1:
                px, py, pz = new_ca[k, 0], new_ca[k, 1], new_ca[k, 2]
            elif src == 2:
                px, py, pz = old_sg[k, 0], old_sg[k, 1], old_sg[k, 2]
            else:
                px, py, pz = new_sg[k, 0], new_sg[k, 1], new_sg[k, 2]
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

    # Compact visited cells into a list so prange can index across them.
    cell_list = np.empty(nc3, dtype=np.int64)
    n_cells = 0
    for c in range(nc3):
        if visited[c] != 0:
            cell_list[n_cells] = c
            n_cells += 1

    moved_first = 2 * (chain_idx * N + ms)
    moved_last = 2 * (chain_idx * N + ms + nm) - 1

    # Per-thread (per-cell) accumulators; reduced at the end.
    e_old_arr = np.zeros(n_cells, dtype=np.float64)
    e_new_arr = np.zeros(n_cells, dtype=np.float64)

    for ci_idx in prange(n_cells):
        c = cell_list[ci_idx]
        cnt = cell_counts[c]
        if cnt == 0: continue
        s = cell_starts[c]
        eo = 0.0; en = 0.0
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
                    dx = ax - old_ca[k, 0]; dy = ay - old_ca[k, 1]; dz = az - old_ca[k, 2]
                    dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                    eo += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                    dx = ax - new_ca[k, 0]; dy = ay - new_ca[k, 1]; dz = az - new_ca[k, 2]
                    dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                    en += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                if not _seq_skip(chain_idx, resi, 1, atom_chain, atom_ri, atom_t,
                                 min_caca, min_casg, min_sgsg):
                    dx = ax - old_sg[k, 0]; dy = ay - old_sg[k, 1]; dz = az - old_sg[k, 2]
                    dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                    eo += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                    dx = ax - new_sg[k, 0]; dy = ay - new_sg[k, 1]; dz = az - new_sg[k, 2]
                    dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                    en += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
        e_old_arr[ci_idx] = eo
        e_new_arr[ci_idx] = en

    return float(e_new_arr.sum() - e_old_arr.sum())
