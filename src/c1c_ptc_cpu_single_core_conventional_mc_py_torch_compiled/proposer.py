"""Segment-batch proposer — torch CPU tensors, with torch.compile wrapping.

Builds B segment proposals in one pass via batched Rodrigues. Calpha and SG
move together as a rigid body. The eager core lives in
`_propose_batch_torch_eager`; `propose_batch_torch` is the public symbol,
optionally bound to a torch.compile'd version of that core.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

_USE_COMPILE = os.environ.get("ROUSE_NO_TORCH_COMPILE", "0") != "1"
_COMPILE_MODE = os.environ.get("ROUSE_TORCH_COMPILE_MODE", "inductor").lower()

try:
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True
except Exception:
    pass


SEG_N_TERMINAL = 0
SEG_C_TERMINAL = 1
SEG_INNER = 2
SEG_BOTH = 3

MTYPE_HINGE = 0
MTYPE_N_TAIL = 1
MTYPE_C_TAIL = 2

AXIS_EPS = 1e-7


def _propose_batch_torch_eager(
        ca_t: torch.Tensor,        # [n_chains, N, 3]
        sg_t: torch.Tensor,
        meta_t: torch.Tensor,      # [B, 6]   (chain, ms, n_moved, seg_type, a_idx, b_idx)
        rand_t: torch.Tensor,      # [B] uniforms in [0, 1)
        max_moved: int,
        N: int, box: float, max_angle: float,
):
    """Return (old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type, valid).

    All output tensors live on the same device/dtype as ca_t.
    Lanes beyond per-row n_moved are zero-padded; consumers must respect
    n_moved_out when indexing.
    """
    device = ca_t.device
    dtype = ca_t.dtype
    B = meta_t.shape[0]
    inv_box = 1.0 / box

    chain_idx = meta_t[:, 0].long()
    ms = meta_t[:, 1].long()
    nm_in = meta_t[:, 2].long()
    seg_type = meta_t[:, 3].long()
    a_idx = meta_t[:, 4].long()
    b_idx = meta_t[:, 5].long()

    is_inner = (seg_type == SEG_INNER)
    is_n_tail = (seg_type == SEG_N_TERMINAL) | (seg_type == SEG_BOTH)

    # k-range + lane mask for padding
    k_range = torch.arange(max_moved, device=device, dtype=torch.long)
    lane = k_range.unsqueeze(0).expand(B, -1)
    lane_valid = lane < nm_in.unsqueeze(1)

    # Gather old positions (Calpha + SG) at index ms+k for each lane.
    bead_idx = (ms.unsqueeze(1) + lane).clamp(0, N - 1)
    chain_e = chain_idx.unsqueeze(1).expand(B, max_moved)
    old_ca = ca_t[chain_e, bead_idx]          # [B, max_moved, 3]
    old_sg = sg_t[chain_e, bead_idx]

    # Compute SG offset relative to Calpha (preserved through MIC).
    sg_off = old_sg - old_ca
    sg_off = sg_off - box * torch.round(sg_off * inv_box)

    # Anchor positions (Calpha) and rotation axis.
    anc = ca_t[chain_idx, a_idx]              # [B, 3]
    pb = ca_t[chain_idx, b_idx]               # [B, 3]

    # Unwrapped Calphas relative to anchor.
    # For inner (a_idx == ms-1) and tail (a_idx == ms+nm or ms-1): use cumulative MIC.
    # We forward-unwrap when a_idx <= ms (inner / c_tail), backward when a_idx > ms (n_tail).
    forward_mask = a_idx <= ms                # [B] bool

    # Build unwrapped Calphas:
    #   forward: uw[k] = anc + cumsum_{i<=k} mic_delta(ca[ms+i-1] -> ca[ms+i]); but uw[0] = anc + mic(anc -> ca[ms]).
    # Use vector form: bonds along the segment + initial mic from anchor to ms-th bead.
    d0_fwd = old_ca[:, 0:1, :] - anc.unsqueeze(1)
    d0_fwd = d0_fwd - box * torch.round(d0_fwd * inv_box)
    bonds = old_ca[:, 1:, :] - old_ca[:, :-1, :]
    bonds = bonds - box * torch.round(bonds * inv_box)
    d_concat = torch.cat([d0_fwd, bonds], dim=1) * lane_valid.unsqueeze(-1).to(dtype)
    uw_fwd = anc.unsqueeze(1) + torch.cumsum(d_concat, dim=1)

    # Backward (n_tail): start at last lane and walk back.
    last_idx = (nm_in - 1).clamp(min=0)                         # [B]
    last_pos = old_ca.gather(
        1, last_idx.view(B, 1, 1).expand(B, 1, 3))              # [B, 1, 3]
    d0_bw = last_pos - anc.unsqueeze(1)
    d0_bw = d0_bw - box * torch.round(d0_bw * inv_box)
    # Walk from anchor through MIC bonds back to bead ms.
    # uw_bw[k] = anc + d0_bw - sum_{j=k}^{nm-2} bonds[j]   for k in [0, nm-1]
    bond_mask = (torch.arange(max_moved - 1, device=device).unsqueeze(0)
                 < (nm_in - 1).unsqueeze(1))
    masked_bonds = bonds * bond_mask.unsqueeze(-1).to(dtype)
    cum_right = masked_bonds.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    pad_zero = torch.zeros(B, 1, 3, device=device, dtype=dtype)
    cum_right_padded = torch.cat([cum_right, pad_zero], dim=1)
    uw_bw = anc.unsqueeze(1) + d0_bw - cum_right_padded

    uw_ca = torch.where(forward_mask.view(B, 1, 1), uw_fwd, uw_bw)

    # Rotation axis: for INNER, axis goes from anchor to unwrapped(b_idx).
    # For tails, axis = MIC(anc -> ca[b_idx]).
    pb_unwrap_inner = uw_ca.gather(
        1, last_idx.view(B, 1, 1).expand(B, 1, 3)).squeeze(1)
    inner_b_off = ca_t[chain_idx, b_idx] - ca_t[chain_idx,
                                                (ms + nm_in - 1).clamp(0, N - 1)]
    inner_b_off = inner_b_off - box * torch.round(inner_b_off * inv_box)
    pb_inner = pb_unwrap_inner + inner_b_off
    axis_inner = pb_inner - anc

    axis_tail = pb - anc
    axis_tail = axis_tail - box * torch.round(axis_tail * inv_box)

    axis = torch.where(is_inner.unsqueeze(1), axis_inner, axis_tail)
    axis_norm = axis.norm(dim=1, keepdim=True).clamp(min=AXIS_EPS)
    valid = axis_norm.squeeze(1) >= AXIS_EPS
    u = axis / axis_norm

    angles = (2.0 * rand_t - 1.0) * max_angle
    cos_a = torch.cos(angles).unsqueeze(1)
    sin_a = torch.sin(angles).unsqueeze(1)
    one_m = 1.0 - cos_a
    ux, uy, uz = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    R = torch.stack([
        torch.cat([one_m * ux * ux + cos_a,
                   one_m * ux * uy - sin_a * uz,
                   one_m * ux * uz + sin_a * uy], dim=1),
        torch.cat([one_m * ux * uy + sin_a * uz,
                   one_m * uy * uy + cos_a,
                   one_m * uy * uz - sin_a * ux], dim=1),
        torch.cat([one_m * ux * uz - sin_a * uy,
                   one_m * uy * uz + sin_a * ux,
                   one_m * uz * uz + cos_a], dim=1),
    ], dim=1)  # [B, 3, 3]

    rel = uw_ca - anc.unsqueeze(1)
    new_ca_unwrapped = anc.unsqueeze(1) + torch.bmm(rel, R.transpose(1, 2))
    new_ca = new_ca_unwrapped - box * torch.floor((new_ca_unwrapped + box / 2.0) * inv_box)

    # SG: rotate the (CA->SG) offset by the same R, then re-attach to new_ca.
    new_sg_off = torch.bmm(sg_off, R.transpose(1, 2))
    new_sg_unwrapped = new_ca_unwrapped + new_sg_off
    new_sg = new_sg_unwrapped - box * torch.floor((new_sg_unwrapped + box / 2.0) * inv_box)

    # Mask invalid: keep old.
    inv_mask = (~valid).view(B, 1, 1)
    new_ca = torch.where(inv_mask, old_ca, new_ca)
    new_sg = torch.where(inv_mask, old_sg, new_sg)
    n_moved_out = torch.where(valid, nm_in, torch.zeros_like(nm_in))

    move_type = torch.zeros(B, dtype=torch.long, device=device)
    move_type = torch.where(is_inner, torch.full_like(move_type, MTYPE_HINGE), move_type)
    move_type = torch.where(is_n_tail & ~is_inner,
                            torch.full_like(move_type, MTYPE_N_TAIL), move_type)
    move_type = torch.where(~is_inner & ~is_n_tail,
                            torch.full_like(move_type, MTYPE_C_TAIL), move_type)

    return old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type


def _resolve_compiled_proposer():
    fn = _propose_batch_torch_eager
    if not _USE_COMPILE or _COMPILE_MODE == "eager":
        return fn
    if _COMPILE_MODE == "script":
        try:
            return torch.jit.script(fn)
        except Exception:
            return fn
    try:
        if _COMPILE_MODE == "aot_eager":
            return torch.compile(fn, backend="aot_eager", fullgraph=False, dynamic=False)
        if _COMPILE_MODE == "max-autotune":
            return torch.compile(fn, mode="max-autotune", fullgraph=False, dynamic=False)
        return torch.compile(fn, mode="reduce-overhead", fullgraph=False, dynamic=False)
    except Exception:
        return fn


propose_batch_torch = _resolve_compiled_proposer()
