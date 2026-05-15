"""Migacz-EMM (Exact Marginalisation of Mutations) multistep MC sweep —
GPU-resident, torch CUDA tensors, CUDA-C proposal kernel.

Per batch:
  1. Build the (B, 6) meta tensor from the round permutation; B <= batch_size.
  2. ONE propose_batch_torch call (hand-written CUDA-C kernel via cupy)
     -> (B, M, 3) old/new ca/sg.
  3. ONE batch_delta_e_torch call -> (B,) delta_e (vs reference state).
  4. ONE correction_matrix_torch call -> (B, B) cross-proposal correction.
  5. Host-side sequential-with-correction Metropolis: each acceptance at i
     updates downstream delta_e[j] += corr[i, j] before j's accept decision.
  6. Scatter accepted moves back into GPU state.

The round-permutation builder ensures no two segments in the same batch
touch the same chain, which is what makes the rank-1 EMM correction valid.

The host-side accept loop is O(B) Python iterations per batch — trivial
vs the GPU work. State remains on GPU throughout; transfers only for the
(B,) delta_e and (B, B) correction at accept time.
"""

from __future__ import annotations

import numpy as np
import torch

from .chain import ChainState, SegmentInfo
from .config import SimConfig
from .energy import (batch_delta_e_cuda, correction_matrix_cuda,
                     causal_accept_cuda)
from .proposer import propose_batch_torch, MTYPE_HINGE, MTYPE_N_TAIL, MTYPE_C_TAIL


_MOVE_NAMES = ("hinge", "n_tail", "c_tail")


CHECKLIST_COUNTERS = {
    "ev_calls": 0,
    "max_disp": 0.0,
    "disp_violations": 0,
}


def _reset_counters():
    CHECKLIST_COUNTERS["ev_calls"] = 0
    CHECKLIST_COUNTERS["max_disp"] = 0.0
    CHECKLIST_COUNTERS["disp_violations"] = 0


class SimulationStats:
    def __init__(self):
        self.attempted = {"hinge": 0, "n_tail": 0, "c_tail": 0}
        self.accepted = {"hinge": 0, "n_tail": 0, "c_tail": 0}

    def record(self, mtype: str, accepted: bool):
        self.attempted[mtype] += 1
        if accepted:
            self.accepted[mtype] += 1

    def add_batch(self, attempted_arr, accepted_arr) -> None:
        for i, name in enumerate(_MOVE_NAMES):
            self.attempted[name] += int(attempted_arr[i])
            self.accepted[name] += int(accepted_arr[i])

    def acceptance(self, mtype: str) -> float:
        a = self.attempted[mtype]
        return self.accepted[mtype] / a if a > 0 else 0.0


def _build_round_permutation(seg_info: SegmentInfo, n_chains: int, rng):
    """Group segments so each round has at most one segment per chain
    (rank-1 EMM correction requires no anchor overlap inside a batch).
    """
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


def perform_sweep(state, cfg: SimConfig, stats: SimulationStats,
                  rng: np.random.Generator, scratch=None):
    """GPU-resident Migacz-EMM multistep sweep.

    `scratch` is accepted for signature compatibility with the conventional
    pair `g1c_cc.perform_sweep`; the multistep path uses the legacy
    free-buffer proposer dispatch which allocates its own per-batch
    buffers. Returns None (no single-kernel timing applies to a
    batch-orchestrated sweep).
    """
    seg_info = state.segments
    N = cfg.N
    box = cfg.box_size
    kBT = cfg.kBT
    M = max(int(seg_info.table[:, 2].max()), 1)
    n_chains = cfg.n_chains
    device = state.ca.device
    dtype = state.ca.dtype

    ca_t = state.ca
    sg_t = state.sg
    table_full = seg_info.table
    inv_kBT = 1.0 / float(kBT)

    perm, boundaries = _build_round_permutation(seg_info, n_chains, rng)
    B_max = min(int(getattr(cfg, "batch_size", 256)), n_chains)

    batch_ranges = []
    for ri in range(len(boundaries) - 1):
        for bs in range(boundaries[ri], boundaries[ri + 1], B_max):
            be = min(bs + B_max, boundaries[ri + 1])
            batch_ranges.append((bs, be))

    # GPU-resident accumulators — read back once at sweep end so the per-batch
    # hot path has zero GPU->host round-trips (propose -> delta_e -> correction
    # -> causal accept are four on-GPU launches; state stays GPU-resident).
    attempted_acc = torch.zeros(3, dtype=torch.int64, device=device)
    accepted_acc = torch.zeros(3, dtype=torch.int64, device=device)
    ev_calls_acc = torch.zeros((), dtype=torch.int64, device=device)
    disp_viol_acc = torch.zeros((), dtype=torch.int64, device=device)
    max_disp_acc = torch.zeros((), dtype=dtype, device=device)

    for bs, be in batch_ranges:
        B = be - bs
        if B == 0:
            continue
        meta_np = table_full[perm[bs:be]]
        meta_t = torch.from_numpy(meta_np).to(device)
        angle_unif = torch.rand(B, device=device, dtype=dtype)

        old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type = propose_batch_torch(
            ca_t, sg_t, meta_t, angle_unif, M, N, box, cfg.max_angle_hinge,
        )
        ci_l = meta_t[:, 0].long()
        bs_l = meta_t[:, 1].long()

        delta_e = batch_delta_e_cuda(
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            ci_l, bs_l, ca_t, sg_t, N, box,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )

        corr = correction_matrix_cuda(
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            ci_l, bs_l, N, box,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )

        # Checklist diagnostics — accumulated on-GPU, read back once at end.
        diff = new_ca - old_ca
        diff = diff - box * torch.round(diff / box)
        disps = diff.norm(dim=-1)
        lane_mask = (torch.arange(M, device=device).unsqueeze(0)
                     < n_moved_out.unsqueeze(1))
        disps = disps * lane_mask.to(disps.dtype)
        ev_calls_acc += (n_moved_out > 0).sum()
        max_disp_acc = torch.maximum(max_disp_acc, disps.max())
        disp_viol_acc += ((disps > cfg.l0 * 1.05) & lane_mask).sum()

        # Host-drawn uniforms (deterministic, seeded); the kernel consumes them
        # densely over proposals with n_moved > 0, matching the legacy loop.
        u_t = torch.from_numpy(rng.random(B).astype(np.float32)).to(device)

        causal_accept_cuda(
            ca_t, sg_t, new_ca, new_sg, delta_e, corr,
            n_moved_out, move_type, ci_l, bs_l, u_t,
            attempted_acc, accepted_acc,
            M, B, N, inv_kBT,
        )

    # Single GPU->host readback for the whole sweep.
    stats.add_batch(attempted_acc.cpu().numpy(), accepted_acc.cpu().numpy())
    CHECKLIST_COUNTERS["ev_calls"] += int(ev_calls_acc.item())
    d_max = float(max_disp_acc.item())
    if d_max > CHECKLIST_COUNTERS["max_disp"]:
        CHECKLIST_COUNTERS["max_disp"] = d_max
    CHECKLIST_COUNTERS["disp_violations"] += int(disp_viol_acc.item())
    return None
