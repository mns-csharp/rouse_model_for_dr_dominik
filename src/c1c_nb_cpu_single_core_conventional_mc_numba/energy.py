"""Energy module — three-zone pair potential, cell-list-accelerated.

Conventional MC variant: only the per-proposal `proposal_delta_e` and
the cell-list builder are needed; no rank-1 correction matrix.

Pair types in the SURPASS-alpha layout:
  - Calpha-Calpha: skip |i-j| < min_seq_caca on same chain
  - Calpha-SG:    skip |i-j| < min_seq_casg
  - SG-SG:        skip |i-j| < min_seq_sgsg

Flat bead layout (interleaved Calpha, SG):
  flat[2*(c*N+r)+0] = ca[c, r]  ;  flat[2*(c*N+r)+1] = sg[c, r]
"""

from __future__ import annotations

import numpy as np
from numba import njit


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


@njit(cache=True, fastmath=False)
def build_cell_list(flat_pos, nc, inv_cs, half_box):
    n = flat_pos.shape[0]
    nc_m1 = nc - 1
    nc3 = nc * nc * nc
    cell_idx = np.empty(n, dtype=np.int64)
    for f in range(n):
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


@njit(cache=True, fastmath=False)
def proposal_delta_e(
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
    moved_first = 2 * (chain_idx * N + ms)
    moved_last = 2 * (chain_idx * N + ms + nm) - 1
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
            ax = flat_pos[atom_f, 0]
            ay = flat_pos[atom_f, 1]
            az = flat_pos[atom_f, 2]
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
                    e_old += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                    dx = ax - new_ca[k, 0]; dy = ay - new_ca[k, 1]; dz = az - new_ca[k, 2]
                    dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                    e_new += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                if not _seq_skip(chain_idx, resi, 1, atom_chain, atom_ri, atom_t,
                                 min_caca, min_casg, min_sgsg):
                    dx = ax - old_sg[k, 0]; dy = ay - old_sg[k, 1]; dz = az - old_sg[k, 2]
                    dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                    e_old += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
                    dx = ax - new_sg[k, 0]; dy = ay - new_sg[k, 1]; dz = az - new_sg[k, 2]
                    dx -= box * round(dx * inv_box); dy -= box * round(dy * inv_box); dz -= box * round(dz * inv_box)
                    e_new += _pair_energy(dx*dx + dy*dy + dz*dz, r_rep_sq, r_max_sq, rep_e, contact_e)
    return e_new - e_old
