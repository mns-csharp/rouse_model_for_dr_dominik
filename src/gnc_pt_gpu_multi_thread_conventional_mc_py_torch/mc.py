"""Segmented conventional MC sweep — torch CUDA tensors, multi-stream.

Per-segment sequential MC has no stream-level parallelism by itself
(each accept depends on the prior state). To extract real parallelism
the multi_thread variant batches each "round" (one segment per chain)
into a single dispatch:

  - Build a round permutation (`_build_round_permutation`) so each round
    contains at most one segment per chain.
  - For each round, slice the round's segments across `cfg._n_streams`
    concurrent CUDA streams; each stream runs propose + delta-E for its
    stripe.
  - After streams join, compute the rank-1 correction matrix so the
    sequential host-side accept matches the bit-by-bit behaviour of
    sequential conventional MC. Within a round, all proposals are on
    distinct chains, so same-chain entries of the correction matrix are
    zero by construction; only cross-chain corrections are non-zero.
  - Apply accepted moves and advance to the next round.

This makes the multi_thread variant numerically equivalent to the
single_thread sibling (modulo the deterministic order in which segments
are emitted within a round). With `cfg._n_streams == 1` the path
collapses to single-stream behaviour exactly.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch

from .chain import ChainState, SegmentInfo
from .config import SimConfig
from .energy import batch_delta_e_torch
from .proposer import propose_batch_torch, MTYPE_HINGE, MTYPE_N_TAIL, MTYPE_C_TAIL


_MOVE_NAMES = ("hinge", "n_tail", "c_tail")


CHECKLIST_COUNTERS = {
    "ev_calls": 0,
    "max_disp": 0.0,
    "disp_violations": 0,
    "rng_sync_stalls": 0,
    "rng_refill_triggered": True,
}


def _reset_counters():
    CHECKLIST_COUNTERS["ev_calls"] = 0
    CHECKLIST_COUNTERS["max_disp"] = 0.0
    CHECKLIST_COUNTERS["disp_violations"] = 0
    CHECKLIST_COUNTERS["rng_sync_stalls"] = 0
    CHECKLIST_COUNTERS["rng_refill_triggered"] = True


class SimulationStats:
    def __init__(self):
        self.attempted = {"hinge": 0, "n_tail": 0, "c_tail": 0}
        self.accepted = {"hinge": 0, "n_tail": 0, "c_tail": 0}

    def record(self, mtype: str, accepted: bool):
        self.attempted[mtype] += 1
        if accepted:
            self.accepted[mtype] += 1

    def acceptance(self, mtype: str) -> float:
        a = self.attempted[mtype]
        return self.accepted[mtype] / a if a > 0 else 0.0


def _build_round_permutation(seg_info: SegmentInfo, n_chains: int, rng):
    """Same round-builder as the multistep variants (one segment per chain per round)."""
    total = seg_info.total_segments
    spc = seg_info.segs_per_chain
    table = seg_info.table
    buckets = [[] for _ in range(n_chains)]
    for gs in range(total):
        buckets[int(table[gs, SegmentInfo.META_CHAIN_IDX])].append(gs)
    for bucket in buckets:
        rng.shuffle(bucket)
    rounds = []
    for k in range(spc):
        round_segs = [b[k] for b in buckets if k < len(b)]
        rng.shuffle(round_segs)
        rounds.append(round_segs)
    perm = []
    boundaries = [0]
    for r in rounds:
        perm.extend(r)
        boundaries.append(len(perm))
    return np.asarray(perm, dtype=np.int64), boundaries


def _even_slices(B: int, n: int) -> List[Tuple[int, int]]:
    if n <= 1 or B == 0:
        return [(0, B)]
    base, rem = divmod(B, n)
    out, s = [], 0
    for w in range(n):
        sz = base + (1 if w < rem else 0)
        if sz == 0:
            continue
        out.append((s, s + sz))
        s += sz
    return out


def _correction_matrix_torch(
        old_ca, new_ca, old_sg, new_sg, n_moved,
        chain_idx, bead_start, N, box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg,
):
    """Dense BxB upper-triangular rank-1 correction. Same form as the multistep
    sibling — copied here so the conventional folder stays self-contained.

    Within a round all proposals are on distinct chains, so same-chain
    sequence-separation skips trivially apply to zero pairs; the matrix is
    typically dense across cross-chain (i, j) entries.
    """
    B = old_ca.shape[0]
    device = old_ca.device
    dtype = old_ca.dtype
    inv_box = 1.0 / box
    corr = torch.zeros(B, B, dtype=dtype, device=device)

    def _zone_e(r2):
        e = torch.where(r2 < r_rep_sq, torch.full_like(r2, rep_e), torch.zeros_like(r2))
        if contact_e != 0.0:
            e = torch.where((r2 >= r_rep_sq) & (r2 < r_max_sq),
                            torch.full_like(r2, contact_e), e)
        return e

    def _pair_e(p1, p2, mask):
        d = p2.unsqueeze(0) - p1.unsqueeze(1)
        d -= box * torch.round(d * inv_box)
        r2 = (d * d).sum(dim=-1)
        return (_zone_e(r2) * mask).sum()

    for i in range(B):
        nm_i = int(n_moved[i].item())
        if nm_i == 0:
            continue
        ci = int(chain_idx[i].item())
        ms_i = int(bead_start[i].item())
        for j in range(i + 1, B):
            nm_j = int(n_moved[j].item())
            if nm_j == 0:
                continue
            cj = int(chain_idx[j].item())
            ms_j = int(bead_start[j].item())
            ri = torch.arange(ms_i, ms_i + nm_i, device=device)
            rj = torch.arange(ms_j, ms_j + nm_j, device=device)
            seq = (ri.unsqueeze(1) - rj.unsqueeze(0)).abs()
            same_chain = (ci == cj)
            zero_mask = torch.zeros_like(seq, dtype=torch.bool)
            skip_caca = (seq < min_caca) if same_chain else zero_mask
            skip_casg = (seq < min_casg) if same_chain else zero_mask
            skip_sgsg = (seq < min_sgsg) if same_chain else zero_mask
            keep_caca = (~skip_caca).to(dtype)
            keep_casg = (~skip_casg).to(dtype)
            keep_sgsg = (~skip_sgsg).to(dtype)

            e00 = _pair_e(old_ca[i, :nm_i], old_ca[j, :nm_j], keep_caca)
            e01 = _pair_e(old_ca[i, :nm_i], new_ca[j, :nm_j], keep_caca)
            e10 = _pair_e(new_ca[i, :nm_i], old_ca[j, :nm_j], keep_caca)
            e11 = _pair_e(new_ca[i, :nm_i], new_ca[j, :nm_j], keep_caca)
            e00 += _pair_e(old_ca[i, :nm_i], old_sg[j, :nm_j], keep_casg)
            e01 += _pair_e(old_ca[i, :nm_i], new_sg[j, :nm_j], keep_casg)
            e10 += _pair_e(new_ca[i, :nm_i], old_sg[j, :nm_j], keep_casg)
            e11 += _pair_e(new_ca[i, :nm_i], new_sg[j, :nm_j], keep_casg)
            e00 += _pair_e(old_sg[i, :nm_i], old_ca[j, :nm_j], keep_casg)
            e01 += _pair_e(old_sg[i, :nm_i], new_ca[j, :nm_j], keep_casg)
            e10 += _pair_e(new_sg[i, :nm_i], old_ca[j, :nm_j], keep_casg)
            e11 += _pair_e(new_sg[i, :nm_i], new_ca[j, :nm_j], keep_casg)
            e00 += _pair_e(old_sg[i, :nm_i], old_sg[j, :nm_j], keep_sgsg)
            e01 += _pair_e(old_sg[i, :nm_i], new_sg[j, :nm_j], keep_sgsg)
            e10 += _pair_e(new_sg[i, :nm_i], old_sg[j, :nm_j], keep_sgsg)
            e11 += _pair_e(new_sg[i, :nm_i], new_sg[j, :nm_j], keep_sgsg)
            corr[i, j] = (e11 - e01) - (e10 - e00)
    return corr


def _fused_accept(delta_e: np.ndarray, correction: np.ndarray,
                  n_moved: np.ndarray, kBT: float, uniforms: np.ndarray):
    """Sequential Metropolis with rank-1 update — runs on host."""
    B = delta_e.shape[0]
    accepted = np.zeros(B, dtype=np.bool_)
    u_idx = 0
    for i in range(B):
        if n_moved[i] == 0:
            continue
        de = float(delta_e[i])
        if de <= 0.0:
            prob = 1.0
        else:
            exponent = -de / kBT
            if exponent <= -745.0:
                prob = 0.0
            elif exponent >= 709.0:
                prob = 1.0
            else:
                prob = math.exp(exponent)
        if uniforms[u_idx] < prob:
            accepted[i] = True
            for j in range(i + 1, B):
                if n_moved[j] == 0:
                    continue
                c = float(correction[i, j])
                if c != 0.0:
                    delta_e[j] = float(delta_e[j]) + c
        u_idx += 1
    return accepted


def perform_sweep(state, cfg: SimConfig, stats: SimulationStats,
                  rng: np.random.Generator) -> None:
    seg_info = state.segments
    device = state.device
    dtype = state.dtype
    table_t = torch.from_numpy(seg_info.table).to(device)
    N = cfg.N
    n_chains = cfg.n_chains
    box = cfg.box_size
    kBT = cfg.kBT
    M = max(seg_info.max_moved_static, 1)

    streams: Sequence[torch.cuda.Stream] = getattr(state, "streams", [])
    n_streams = len(streams)

    perm, boundaries = _build_round_permutation(seg_info, n_chains, rng)

    # Per-segment B=1 dispatch within rounds — matches multistep at batch_size=1
    # exactly so the round-permutation order + RNG-draw shape are identical
    # to the multistep code path (cluster E in the bit-exact audit).
    batch_ranges = []
    for ri in range(len(boundaries) - 1):
        for bs in range(boundaries[ri], boundaries[ri + 1]):
            batch_ranges.append((bs, bs + 1))

    for bs, be in batch_ranges:
        B = be - bs  # always 1
        meta_t = table_t[torch.from_numpy(perm[bs:be]).to(device)]
        rand_t = torch.from_numpy(rng.random(B).astype(np.float32)).to(device)

        ca_t = state.ca
        sg_t = state.sg

        old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type = propose_batch_torch(
            ca_t, sg_t, meta_t, rand_t, M, N, box, cfg.max_angle_hinge,
        )
        delta_e = batch_delta_e_torch(
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            meta_t[:, 0].long(), meta_t[:, 1].long(),
            ca_t, sg_t,
            N, box,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )

        corr = _correction_matrix_torch(
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            meta_t[:, 0].long(), meta_t[:, 1].long(), N, box,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )

        de_np = delta_e.detach().cpu().numpy().astype(np.float64).copy()
        corr_np = corr.detach().cpu().numpy().astype(np.float64).copy()
        nm_np = n_moved_out.detach().cpu().numpy().astype(np.int64)
        u_np = rng.random(B).astype(np.float64)
        accepted = _fused_accept(de_np, corr_np, nm_np, kBT, u_np)

        CHECKLIST_COUNTERS["ev_calls"] += int(B)
        diffs = (new_ca - old_ca)
        diffs = diffs - cfg.box_size * torch.round(diffs / cfg.box_size)
        disps_all = diffs.norm(dim=-1)
        if int(disps_all.numel()) > 0:
            d_max = float(disps_all.max().item())
            if d_max > CHECKLIST_COUNTERS["max_disp"]:
                CHECKLIST_COUNTERS["max_disp"] = d_max
            n_viol = int((disps_all > cfg.l0 * 1.05).sum().item())
            CHECKLIST_COUNTERS["disp_violations"] += n_viol

        for i in range(B):
            nm = int(nm_np[i])
            mtype = _MOVE_NAMES[int(move_type[i].item())]
            stats.record(mtype, bool(accepted[i]))
            if nm == 0 or not accepted[i]:
                continue
            ci = int(meta_t[i, 0].item())
            ms = int(meta_t[i, 1].item())
            state.ca[ci, ms:ms + nm, :] = new_ca[i, :nm]
            state.sg[ci, ms:ms + nm, :] = new_sg[i, :nm]


def _propose_and_delta_e_streamed(
        streams: Sequence[torch.cuda.Stream],
        ca_t: torch.Tensor, sg_t: torch.Tensor,
        meta_t: torch.Tensor, rand_t: torch.Tensor,
        max_moved: int, N: int, cfg: SimConfig, B: int):
    """Run propose + delta-E for a round, sliced across CUDA streams.

    Mirrors the multistep multi_thread streaming path: identical join
    semantics so all torch state is back on the default stream by return.
    """
    device = ca_t.device
    dtype = ca_t.dtype
    M = max(int(max_moved), 1)
    n_streams = len(streams)
    slices = _even_slices(B, n_streams)

    old_ca_full = torch.empty(B, M, 3, dtype=dtype, device=device)
    new_ca_full = torch.empty(B, M, 3, dtype=dtype, device=device)
    old_sg_full = torch.empty(B, M, 3, dtype=dtype, device=device)
    new_sg_full = torch.empty(B, M, 3, dtype=dtype, device=device)
    n_moved_full = torch.empty(B, dtype=torch.int64, device=device)
    move_type_full = torch.empty(B, dtype=torch.int64, device=device)
    delta_e_full = torch.empty(B, dtype=dtype, device=device)

    default = torch.cuda.current_stream(device)
    used_streams = []
    for k, (bs, be) in enumerate(slices):
        s = streams[k % n_streams]
        used_streams.append(s)
        s.wait_stream(default)
        with torch.cuda.stream(s):
            meta_s = meta_t[bs:be]
            rand_s = rand_t[bs:be]
            old_ca_s, new_ca_s, old_sg_s, new_sg_s, nm_s, mt_s = propose_batch_torch(
                ca_t, sg_t, meta_s, rand_s, M, N,
                cfg.box_size, cfg.max_angle_hinge,
            )
            old_ca_full[bs:be] = old_ca_s
            new_ca_full[bs:be] = new_ca_s
            old_sg_full[bs:be] = old_sg_s
            new_sg_full[bs:be] = new_sg_s
            n_moved_full[bs:be] = nm_s
            move_type_full[bs:be] = mt_s

            de_s = batch_delta_e_torch(
                old_ca_s, new_ca_s, old_sg_s, new_sg_s, nm_s,
                meta_s[:, 0].long(), meta_s[:, 1].long(),
                ca_t, sg_t,
                N, cfg.box_size,
                cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
                cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
            )
            delta_e_full[bs:be] = de_s

    for s in used_streams:
        default.wait_stream(s)

    return (old_ca_full, new_ca_full, old_sg_full, new_sg_full,
            n_moved_full, move_type_full, delta_e_full)
