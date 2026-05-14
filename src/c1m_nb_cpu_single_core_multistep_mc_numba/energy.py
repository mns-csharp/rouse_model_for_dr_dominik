"""Energy module — 3-zone repulsive/contact pair potential, cell-list-accelerated.

Pair types in the SURPASS-alpha layout:
  - Calpha-Calpha: rigid backbone, sequence-separation skip |i-j| < min_seq_caca
  - Calpha-SG:    skip |i-j| < min_seq_casg (intra-residue)
  - SG-SG:        skip |i-j| < min_seq_sgsg

Flat bead layout:
  flat index f = 2 * (chain * N + residue) + bead_type     (0 = Calpha, 1 = SG)
  bead_type for flat index f = f & 1
  residue   for flat index f = f >> 1
  chain_idx for flat index f = (f >> 1) // N

Energy zones (per pair):
  r < r_rep      : penalty = repulsive_energy
  r_rep <= r < r_max : 0   (no zone-2 attractive well in athermal default)
  r >= r_max         : 0
  if contact_energy != 0: zone (r_min..r_max) = contact_energy.
For the simple athermal model used here we just use r_rep_sq + repulsive_energy.

For multistep MC we compute:
  - delta_e[B]            — energy change of each individual proposal
  - correction[B, B]      — rank-1 correction matrix
                            E[j] += correction[i, j] when proposal i is accepted.
                            Correction satisfies the Migacz identity
                            (1)+(0)-(0,1)-(1,0).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _pair_energy(r2, r_rep_sq, r_max_sq, rep_e, contact_e):
    """Three-zone scalar pair energy."""
    if r2 < r_rep_sq:
        return rep_e
    if contact_e != 0.0 and r2 < r_max_sq:
        return contact_e
    return 0.0


@njit(cache=True, fastmath=False)
def _seq_skip(ci, ri, ti, cj, rj, tj,
              min_caca, min_casg, min_sgsg):
    """Return True if pair (ci, ri, ti) <-> (cj, rj, tj) should be skipped
    due to sequence separation. ti, tj are bead types (0=Calpha, 1=SG)."""
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
    """Bin all beads into cells. Returns (sorted_order, cell_starts, cell_counts)."""
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
    """Compute (E_new - E_old) for one proposal. The proposal moves both
    Calphas and SGs in residues [ms, ms+nm) of chain `chain_idx`."""
    if nm == 0:
        return 0.0
    nc_m1 = nc - 1
    nc3 = nc * nc * nc

    # Visited cells = union of 27-neighborhoods of every (old/new, Calpha/SG) bead.
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

    moved_first_flat = 2 * (chain_idx * N + ms)
    moved_last_flat = 2 * (chain_idx * N + ms + nm) - 1  # inclusive

    e_old = 0.0
    e_new = 0.0
    for c in range(nc3):
        if visited[c] == 0: continue
        cnt = cell_counts[c]
        if cnt == 0: continue
        s = cell_starts[c]
        for ii in range(cnt):
            atom_f = sorted_order[s + ii]
            # Skip beads inside the moved region (they live in old_/new_ buffers).
            if atom_f >= moved_first_flat and atom_f <= moved_last_flat:
                continue
            ax = flat_pos[atom_f, 0]
            ay = flat_pos[atom_f, 1]
            az = flat_pos[atom_f, 2]
            atom_t = atom_f & 1
            atom_res = atom_f >> 1
            atom_chain = atom_res // N
            atom_ri = atom_res - atom_chain * N
            for k in range(nm):
                # moved Calpha
                resi_ca = ms + k
                if not _seq_skip(chain_idx, resi_ca, 0,
                                 atom_chain, atom_ri, atom_t,
                                 min_caca, min_casg, min_sgsg):
                    dx = ax - old_ca[k, 0]
                    dy = ay - old_ca[k, 1]
                    dz = az - old_ca[k, 2]
                    dx -= box * round(dx * inv_box)
                    dy -= box * round(dy * inv_box)
                    dz -= box * round(dz * inv_box)
                    e_old += _pair_energy(dx*dx + dy*dy + dz*dz,
                                          r_rep_sq, r_max_sq, rep_e, contact_e)
                    dx = ax - new_ca[k, 0]
                    dy = ay - new_ca[k, 1]
                    dz = az - new_ca[k, 2]
                    dx -= box * round(dx * inv_box)
                    dy -= box * round(dy * inv_box)
                    dz -= box * round(dz * inv_box)
                    e_new += _pair_energy(dx*dx + dy*dy + dz*dz,
                                          r_rep_sq, r_max_sq, rep_e, contact_e)
                # moved SG
                if not _seq_skip(chain_idx, resi_ca, 1,
                                 atom_chain, atom_ri, atom_t,
                                 min_caca, min_casg, min_sgsg):
                    dx = ax - old_sg[k, 0]
                    dy = ay - old_sg[k, 1]
                    dz = az - old_sg[k, 2]
                    dx -= box * round(dx * inv_box)
                    dy -= box * round(dy * inv_box)
                    dz -= box * round(dz * inv_box)
                    e_old += _pair_energy(dx*dx + dy*dy + dz*dz,
                                          r_rep_sq, r_max_sq, rep_e, contact_e)
                    dx = ax - new_sg[k, 0]
                    dy = ay - new_sg[k, 1]
                    dz = az - new_sg[k, 2]
                    dx -= box * round(dx * inv_box)
                    dy -= box * round(dy * inv_box)
                    dz -= box * round(dz * inv_box)
                    e_new += _pair_energy(dx*dx + dy*dy + dz*dz,
                                          r_rep_sq, r_max_sq, rep_e, contact_e)
    return e_new - e_old


@njit(cache=True, fastmath=False)
def batch_delta_e(
        B, old_ca, new_ca, old_sg, new_sg, n_moved,
        chain_idx_arr, bead_start_arr,
        N, flat_pos,
        nc, inv_cs, half_box, neighbor_offsets,
        sorted_order, cell_starts, cell_counts,
        box, inv_box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg):
    out = np.zeros(B, dtype=np.float64)
    for b in range(B):
        nm = n_moved[b]
        if nm == 0:
            continue
        out[b] = proposal_delta_e(
            old_ca[b], new_ca[b], old_sg[b], new_sg[b], nm,
            chain_idx_arr[b], bead_start_arr[b], N,
            flat_pos,
            nc, inv_cs, half_box, neighbor_offsets,
            sorted_order, cell_starts, cell_counts,
            box, inv_box,
            r_rep_sq, r_max_sq, rep_e, contact_e,
            min_caca, min_casg, min_sgsg)
    return out


@njit(cache=True, fastmath=False)
def correction_matrix(
        B,
        old_ca, new_ca, old_sg, new_sg, n_moved,
        chain_idx_arr, bead_start_arr, N,
        box, inv_box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg):
    """Build the upper-triangular Migacz rank-1 correction matrix.

    correction[i, j] (i<j) = E11 + E00 - E01 - E10  for the moved beads
    of proposals i and j (each proposal contributes 2*nm beads: nm Calphas + nm SGs).

    Sign convention (matches fused_accept): when proposal i is accepted,
    delta_e[j] += correction[i, j] for all j > i.
    The contribution accounts for the swap of i's old beads with i's new
    beads as evaluated against j's beads (whose move state we still
    consider — old vs new).
    """
    corr = np.zeros((B, B), dtype=np.float64)
    for i in range(B):
        nm_i = n_moved[i]
        if nm_i == 0: continue
        ci = chain_idx_arr[i]
        ms_i = bead_start_arr[i]
        for j in range(i + 1, B):
            nm_j = n_moved[j]
            if nm_j == 0: continue
            cj = chain_idx_arr[j]
            ms_j = bead_start_arr[j]
            e00 = 0.0; e01 = 0.0; e10 = 0.0; e11 = 0.0
            # Each moved bead in i and each moved bead in j: 2*nm_i x 2*nm_j pairs
            # (Calpha + SG of i  vs  Calpha + SG of j).
            for ai in range(nm_i):
                resi = ms_i + ai
                for ti in range(2):  # 0 = Calpha, 1 = SG
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
                            # E00: i-old vs j-old
                            dx = ojx - oix; dy = ojy - oiy; dz = ojz - oiz
                            dx -= box * round(dx * inv_box)
                            dy -= box * round(dy * inv_box)
                            dz -= box * round(dz * inv_box)
                            e00 += _pair_energy(dx*dx + dy*dy + dz*dz,
                                                r_rep_sq, r_max_sq, rep_e, contact_e)
                            # E01: i-old vs j-new
                            dx = njx - oix; dy = njy - oiy; dz = njz - oiz
                            dx -= box * round(dx * inv_box)
                            dy -= box * round(dy * inv_box)
                            dz -= box * round(dz * inv_box)
                            e01 += _pair_energy(dx*dx + dy*dy + dz*dz,
                                                r_rep_sq, r_max_sq, rep_e, contact_e)
                            # E10: i-new vs j-old
                            dx = ojx - nix; dy = ojy - niy; dz = ojz - niz
                            dx -= box * round(dx * inv_box)
                            dy -= box * round(dy * inv_box)
                            dz -= box * round(dz * inv_box)
                            e10 += _pair_energy(dx*dx + dy*dy + dz*dz,
                                                r_rep_sq, r_max_sq, rep_e, contact_e)
                            # E11: i-new vs j-new
                            dx = njx - nix; dy = njy - niy; dz = njz - niz
                            dx -= box * round(dx * inv_box)
                            dy -= box * round(dy * inv_box)
                            dz -= box * round(dz * inv_box)
                            e11 += _pair_energy(dx*dx + dy*dy + dz*dz,
                                                r_rep_sq, r_max_sq, rep_e, contact_e)
            delta = (e11 - e01) - (e10 - e00)
            if delta != 0.0:
                corr[i, j] = delta
    return corr
