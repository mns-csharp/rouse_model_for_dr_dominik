"""Fused-batched MC sweep with cell-list energy + CUDA streams.

Combines:
  - Round-based parallel-Gibbs design from `g1c_ptgcl/mc.py`.
  - Stream-slice dispatch from `gnc_ptg/mc.py`: the (n_chains,) per-round
    meta tensor is split into `n_streams` slices that run concurrently.

Per round:
  1. Build CA/SG cell list on the DEFAULT stream (single full-state op).
  2. Streams fan out; each stream gathers candidate neighbours for its
     slice and runs propose + cell-list-bounded delta-E.
  3. Streams rejoin the default stream.
  4. Accept/scatter on the default stream over the full batch.

Streams only READ the cell-list arrays; the accept/scatter on the default
stream serialises writes back to state.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .chain import ChainState, SegmentInfo
from .config import SimConfig
from .energy import batch_delta_e_torch_cell_list_fused
from .proposer import propose_batch_torch, MTYPE_HINGE, MTYPE_N_TAIL, MTYPE_C_TAIL
from . import cell_list as _cl


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


def _even_slices(n: int, k: int):
    base, rem = divmod(n, k)
    s = 0
    for i in range(k):
        e = s + base + (1 if i < rem else 0)
        if e > s:
            yield s, e
        s = e


def _propose_delta_e_slice(meta_slice, ca_t, sg_t, M, N, box, cfg, dtype, device,
                            idx_ca, starts_ca, idx_sg, starts_sg, n_cells,
                            max_neighbors):
    Bs = meta_slice.shape[0]
    angle_unif = torch.rand(Bs, device=device, dtype=dtype)
    old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type = propose_batch_torch(
        ca_t, sg_t, meta_slice, angle_unif, M, N, box, cfg.max_angle_hinge,
    )
    BsM = Bs * M
    cand_ca_old = _cl.gather_candidates(old_ca.reshape(BsM, 3), idx_ca, starts_ca,
                                        n_cells, box, max_neighbors)
    cand_ca_new = _cl.gather_candidates(new_ca.reshape(BsM, 3), idx_ca, starts_ca,
                                        n_cells, box, max_neighbors)
    cand_sg_old = _cl.gather_candidates(old_sg.reshape(BsM, 3), idx_sg, starts_sg,
                                        n_cells, box, max_neighbors)
    cand_sg_new = _cl.gather_candidates(new_sg.reshape(BsM, 3), idx_sg, starts_sg,
                                        n_cells, box, max_neighbors)
    delta_e = batch_delta_e_torch_cell_list_fused(
        old_ca, new_ca, old_sg, new_sg, n_moved_out,
        meta_slice[:, 0].long(), meta_slice[:, 1].long(),
        ca_t, sg_t,
        cand_ca_old, cand_ca_new, cand_sg_old, cand_sg_new,
        N, box,
        cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
        cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
    )
    return old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type, delta_e


def _run_round(meta_round_t, ca_t, sg_t, M, N, box, kBT, cfg, stats,
               device, streams, r_cell, max_neighbors):
    B = meta_round_t.shape[0]
    dtype = ca_t.dtype
    n_streams = len(streams)

    flat_ca = ca_t.reshape(-1, 3)
    flat_sg = sg_t.reshape(-1, 3)
    idx_ca, starts_ca, n_cells = _cl.build_cell_index(flat_ca, box, r_cell)
    idx_sg, starts_sg, _ = _cl.build_cell_index(flat_sg, box, r_cell)

    if n_streams >= 2 and B >= 2 * n_streams:
        old_ca = torch.empty(B, M, 3, device=device, dtype=dtype)
        new_ca = torch.empty(B, M, 3, device=device, dtype=dtype)
        old_sg = torch.empty(B, M, 3, device=device, dtype=dtype)
        new_sg = torch.empty(B, M, 3, device=device, dtype=dtype)
        n_moved_out = torch.empty(B, device=device, dtype=torch.long)
        move_type = torch.empty(B, device=device, dtype=torch.long)
        delta_e = torch.empty(B, device=device, dtype=dtype)

        default = torch.cuda.current_stream(device=device)
        slices = list(_even_slices(B, n_streams))
        for stream, (s, e) in zip(streams, slices):
            stream.wait_stream(default)
            with torch.cuda.stream(stream):
                oc, nc, os_, ns_, nm, mt, de = _propose_delta_e_slice(
                    meta_round_t[s:e], ca_t, sg_t, M, N, box, cfg, dtype, device,
                    idx_ca, starts_ca, idx_sg, starts_sg, n_cells, max_neighbors,
                )
                old_ca[s:e].copy_(oc, non_blocking=True)
                new_ca[s:e].copy_(nc, non_blocking=True)
                old_sg[s:e].copy_(os_, non_blocking=True)
                new_sg[s:e].copy_(ns_, non_blocking=True)
                n_moved_out[s:e].copy_(nm, non_blocking=True)
                move_type[s:e].copy_(mt, non_blocking=True)
                delta_e[s:e].copy_(de, non_blocking=True)
        for stream, _se in zip(streams, slices):
            default.wait_stream(stream)
    else:
        old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type, delta_e = \
            _propose_delta_e_slice(
                meta_round_t, ca_t, sg_t, M, N, box, cfg, dtype, device,
                idx_ca, starts_ca, idx_sg, starts_sg, n_cells, max_neighbors,
            )

    valid_lane = (n_moved_out > 0)
    metro_unif = torch.rand(B, device=device, dtype=dtype)

    diff = new_ca - old_ca
    diff = diff - box * torch.round(diff / box)
    disps = diff.norm(dim=-1)
    lane_mask = (torch.arange(M, device=device).unsqueeze(0)
                 < n_moved_out.unsqueeze(1))
    disps = disps * lane_mask.to(disps.dtype)

    exponent = -delta_e / kBT
    exp_clamped = exponent.clamp(min=-745.0, max=0.0)
    accept_prob = torch.where(delta_e <= 0,
                              torch.ones_like(delta_e),
                              torch.exp(exp_clamped))
    accept = (metro_unif < accept_prob) & valid_lane

    mt_h = (move_type == MTYPE_HINGE) & valid_lane
    mt_n = (move_type == MTYPE_N_TAIL) & valid_lane
    mt_c = (move_type == MTYPE_C_TAIL) & valid_lane
    att = torch.stack([mt_h.sum(), mt_n.sum(), mt_c.sum()]).to(torch.long)
    acc = torch.stack([(mt_h & accept).sum(),
                       (mt_n & accept).sum(),
                       (mt_c & accept).sum()]).to(torch.long)
    ev_calls = int(valid_lane.sum().item())
    d_max = float(disps.max().item()) if B > 0 else 0.0
    n_viol = int(((disps > cfg.l0 * 1.05) & lane_mask).sum().item())

    CHECKLIST_COUNTERS["ev_calls"] += ev_calls
    if d_max > CHECKLIST_COUNTERS["max_disp"]:
        CHECKLIST_COUNTERS["max_disp"] = d_max
    CHECKLIST_COUNTERS["disp_violations"] += n_viol
    stats.add_batch(att.cpu().numpy(), acc.cpu().numpy())

    if bool(accept.any().item()):
        accept_idx = accept.nonzero(as_tuple=False).squeeze(-1)
        ci_t = meta_round_t[accept_idx, 0].long()
        ms_t = meta_round_t[accept_idx, 1].long()
        nm_t = n_moved_out[accept_idx].long()
        k_idx = torch.arange(M, device=device).unsqueeze(0)
        res_idx = ms_t.unsqueeze(1) + k_idx
        lane_ok = k_idx < nm_t.unsqueeze(1)
        chain_flat = ci_t.unsqueeze(1).expand_as(res_idx).reshape(-1)
        res_flat = res_idx.reshape(-1)
        new_ca_flat = new_ca[accept_idx].reshape(-1, 3)
        new_sg_flat = new_sg[accept_idx].reshape(-1, 3)
        lane_flat = lane_ok.reshape(-1)

        valid_pos = lane_flat.nonzero(as_tuple=False).squeeze(-1)
        chain_sel = chain_flat[valid_pos]
        res_sel = res_flat[valid_pos]
        ca_sel = new_ca_flat[valid_pos]
        sg_sel = new_sg_flat[valid_pos]

        ca_t.index_put_((chain_sel, res_sel), ca_sel, accumulate=False)
        sg_t.index_put_((chain_sel, res_sel), sg_sel, accumulate=False)


def perform_sweep(state, cfg: SimConfig, stats: SimulationStats,
                  rng: np.random.Generator) -> None:
    seg_info = state.segments
    N = cfg.N
    box = cfg.box_size
    kBT = cfg.kBT
    M = max(int(seg_info.table[:, 2].max()), 1)
    n_chains = state.cfg.n_chains
    segs_per_chain = seg_info.segs_per_chain
    device = state.device
    streams = getattr(state, "streams", [])

    ca_t = state.ca
    sg_t = state.sg
    table_full = seg_info.table

    r_cell = max(float(cfg.r_max), float(cfg.sigma))
    max_neighbors = int(getattr(cfg, "cell_max_neighbors", 256))

    chain_order = np.arange(n_chains, dtype=np.int64)
    rng.shuffle(chain_order)
    round_order = np.arange(segs_per_chain, dtype=np.int64)
    rng.shuffle(round_order)

    for s in round_order:
        rows = chain_order * segs_per_chain + int(s)
        meta_round_np = table_full[rows]
        meta_round_t = torch.from_numpy(meta_round_np).to(device)
        _run_round(meta_round_t, ca_t, sg_t, M, N, box, kBT, cfg, stats,
                   device, streams, r_cell, max_neighbors)

    return None
