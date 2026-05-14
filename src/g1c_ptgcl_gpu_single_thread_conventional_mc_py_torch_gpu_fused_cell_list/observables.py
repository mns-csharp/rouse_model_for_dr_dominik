"""Static observables — Rg^2, R_e^2, end-to-end distance.

Operate on (Calpha-only) unwrapped chains. SG beads are not part of the
backbone observables but are used for the energy.

`total_energy` streams over row chunks of the (NK, NK) pair-energy matrix so
peak transient memory is bounded by `MEMORY_BUDGET_BYTES` (~16 MB) instead
of growing as O((NK)^2). At K=1000 N=25 the dense path peaked > 25 GB and
crashed with `_ArrayMemoryError`; the streaming path holds < 16 MB at any
K we have tested.
"""

from __future__ import annotations

import math

import numpy as np

from .number_space import NumberSpace


# Peak intermediate-allocation budget for the streaming total_energy pass.
# A chunk holds up to ~4 row-chunked tensors of shape (chunk_rows, NK):
# the per-bead r2 matrix, its zone-energy, the keep mask, and one product.
# Picking 16 MiB keeps peak transient working set well within L3 cache for
# the typical (n_chains, N) range we run, and a few hundred MB even at
# extreme K=1000 N=100.
MEMORY_BUDGET_BYTES = 16 * 1024 * 1024


def unwrap_chains(positions: np.ndarray, ns: NumberSpace) -> np.ndarray:
    """Unwrap each chain by cumulative MIC bonds, anchored at bead 0.

    positions: float64[n_chains, N, 3]  (PBC-wrapped).
    Returns:    float64[n_chains, N, 3]  (anchored unwrapped).
    """
    box = ns.box_size
    inv_box = 1.0 / box
    bonds = positions[:, 1:, :] - positions[:, :-1, :]
    bonds = bonds - box * np.round(bonds * inv_box)
    cum = np.cumsum(bonds, axis=1)
    out = np.empty_like(positions)
    out[:, 0, :] = positions[:, 0, :]
    out[:, 1:, :] = positions[:, 0:1, :] + cum
    return out


def radius_of_gyration_sq(unwrapped: np.ndarray) -> np.ndarray:
    """Per-chain Rg^2."""
    com = unwrapped.mean(axis=1, keepdims=True)
    diffs = unwrapped - com
    return (diffs * diffs).sum(axis=(1, 2)) / unwrapped.shape[1]


def end_to_end_sq(unwrapped: np.ndarray) -> np.ndarray:
    d = unwrapped[:, -1, :] - unwrapped[:, 0, :]
    return (d * d).sum(axis=-1)


def chain_com(unwrapped: np.ndarray) -> np.ndarray:
    return unwrapped.mean(axis=1)


def _chunk_rows_for(n_total: int, elem_bytes: int) -> int:
    """Number of rows to keep in flight so a single chunk's intermediates
    (4 buffers of (chunk_rows, n_total) shape) stay under MEMORY_BUDGET_BYTES.
    """
    if n_total <= 0:
        return 1
    per_row_bytes = max(1, n_total * elem_bytes * 4)
    chunk = MEMORY_BUDGET_BYTES // per_row_bytes
    return max(1, min(chunk, n_total))


def total_energy(state, cfg) -> float:
    """Total system energy under the 3-zone pair potential.

    Sums repulsive contributions (r^2 < r_rep_sq) and contact contributions
    (r_rep_sq <= r^2 < r_max_sq) over all CA-CA, CA-SG, and SG-SG pairs,
    skipping intra-chain short-range pairs per cfg.min_seq_caca / min_seq_casg
    / min_seq_sgsg, and excluding the diagonal.

    Streams over row chunks of the (NK, NK) pair matrix; peak transient
    memory bounded by `MEMORY_BUDGET_BYTES`. Symmetric pair types (CA-CA,
    SG-SG) accumulate the full matrix sum and apply the *0.5 outside the
    loop — chunking traverses the full matrix once so double-counting is
    preserved exactly.

    Operates on torch CUDA tensors when state.ca is one (no host transfer).
    """
    import torch

    box = float(cfg.box_size)
    inv_box = 1.0 / box
    rep_e = float(cfg.repulsive_energy)
    contact_e = float(cfg.contact_energy)
    r_rep_sq = float(cfg.r_rep_sq)
    r_max_sq = float(cfg.r_max_sq)
    min_caca = int(cfg.min_seq_caca)
    min_casg = int(cfg.min_seq_casg)
    min_sgsg = int(cfg.min_seq_sgsg)
    has_contact = contact_e != 0.0

    if isinstance(state.ca, torch.Tensor):
        device = state.ca.device
        dtype = torch.float32
        ca = state.ca.contiguous().view(-1, 3).to(dtype)
        sg = state.sg.contiguous().view(-1, 3).to(dtype)
        n_total = ca.shape[0]
        N = int(cfg.N)

        idx_full = torch.arange(n_total, device=device)
        chain_full = idx_full // N
        resid_full = idx_full % N

        chunk_rows = _chunk_rows_for(n_total, elem_bytes=4)

        def zone(r2):
            e = torch.where(r2 < r_rep_sq,
                            torch.full_like(r2, rep_e),
                            torch.zeros_like(r2))
            if has_contact:
                e = torch.where((r2 >= r_rep_sq) & (r2 < r_max_sq),
                                torch.full_like(r2, contact_e), e)
            return e

        e_caca = 0.0
        e_casg = 0.0
        e_sgsg = 0.0
        for cs in range(0, n_total, chunk_rows):
            ce = min(cs + chunk_rows, n_total)
            ca_chunk = ca[cs:ce]                                  # (chunk, 3)
            sg_chunk = sg[cs:ce]
            chain_chunk = chain_full[cs:ce]                        # (chunk,)
            resid_chunk = resid_full[cs:ce]

            same_chain = chain_chunk.unsqueeze(1) == chain_full.unsqueeze(0)
            seq_diff = (resid_chunk.unsqueeze(1) - resid_full.unsqueeze(0)).abs()
            row_idx = torch.arange(cs, ce, device=device).unsqueeze(1)
            eye_rows = row_idx == idx_full.unsqueeze(0)

            keep_caca = (~(same_chain & (seq_diff < min_caca))) & ~eye_rows
            keep_casg = ~(same_chain & (seq_diff < min_casg))
            keep_sgsg = (~(same_chain & (seq_diff < min_sgsg))) & ~eye_rows

            d = ca.unsqueeze(0) - ca_chunk.unsqueeze(1)            # (chunk, NK, 3)
            d = d - box * torch.round(d * inv_box)
            r2_caca = (d * d).sum(dim=-1)
            d = sg.unsqueeze(0) - ca_chunk.unsqueeze(1)
            d = d - box * torch.round(d * inv_box)
            r2_casg = (d * d).sum(dim=-1)
            d = sg.unsqueeze(0) - sg_chunk.unsqueeze(1)
            d = d - box * torch.round(d * inv_box)
            r2_sgsg = (d * d).sum(dim=-1)

            e_caca += float((zone(r2_caca) * keep_caca).sum().item())
            e_casg += float((zone(r2_casg) * keep_casg).sum().item())
            e_sgsg += float((zone(r2_sgsg) * keep_sgsg).sum().item())

        e_caca *= 0.5
        e_sgsg *= 0.5
        return float(e_caca + e_casg + e_sgsg)

    # ---- numpy fallback (CPU apps where state.ca is np.ndarray) ----
    ca = state.ca.reshape(-1, 3)
    sg = state.sg.reshape(-1, 3)
    n_total = ca.shape[0]
    N = int(cfg.N)
    idx_full = np.arange(n_total)
    chain_full = idx_full // N
    resid_full = idx_full % N

    elem_bytes = ca.dtype.itemsize
    chunk_rows = _chunk_rows_for(n_total, elem_bytes=elem_bytes)

    def zone_np(r2):
        e = np.where(r2 < r_rep_sq, rep_e, 0.0)
        if has_contact:
            e = np.where((r2 >= r_rep_sq) & (r2 < r_max_sq), contact_e, e)
        return e

    e_caca = 0.0
    e_casg = 0.0
    e_sgsg = 0.0
    for cs in range(0, n_total, chunk_rows):
        ce = min(cs + chunk_rows, n_total)
        ca_chunk = ca[cs:ce]
        sg_chunk = sg[cs:ce]
        chain_chunk = chain_full[cs:ce]
        resid_chunk = resid_full[cs:ce]

        same_chain = chain_chunk[:, None] == chain_full[None, :]
        seq_diff = np.abs(resid_chunk[:, None] - resid_full[None, :])
        row_idx = np.arange(cs, ce)[:, None]
        eye_rows = row_idx == idx_full[None, :]

        keep_caca = (~(same_chain & (seq_diff < min_caca))) & ~eye_rows
        keep_casg = ~(same_chain & (seq_diff < min_casg))
        keep_sgsg = (~(same_chain & (seq_diff < min_sgsg))) & ~eye_rows

        d = ca[None, :, :] - ca_chunk[:, None, :]                  # (chunk, NK, 3)
        d = d - box * np.round(d * inv_box)
        r2_caca = (d * d).sum(axis=-1)
        d = sg[None, :, :] - ca_chunk[:, None, :]
        d = d - box * np.round(d * inv_box)
        r2_casg = (d * d).sum(axis=-1)
        d = sg[None, :, :] - sg_chunk[:, None, :]
        d = d - box * np.round(d * inv_box)
        r2_sgsg = (d * d).sum(axis=-1)

        e_caca += float((zone_np(r2_caca) * keep_caca).sum())
        e_casg += float((zone_np(r2_casg) * keep_casg).sum())
        e_sgsg += float((zone_np(r2_sgsg) * keep_sgsg).sum())

    e_caca *= 0.5
    e_sgsg *= 0.5
    return float(e_caca + e_casg + e_sgsg)


def summary(state) -> dict:
    """Return a dict of mean/std for {Rg^2, R_e^2}.

    state.ca is a torch CUDA tensor here — pull to CPU and convert to numpy
    once per observable readout. The cost is amortized across `sample_interval`
    sweeps so it's negligible vs. the MC inner loop.
    """
    import torch
    if isinstance(state.ca, torch.Tensor):
        ca_np = state.ca.detach().cpu().numpy().astype(__import__("numpy").float64)
    else:
        ca_np = state.ca
    uw = unwrap_chains(ca_np, state.ns)
    rg2 = radius_of_gyration_sq(uw)
    re2 = end_to_end_sq(uw)
    return {
        "rg2_mean": float(rg2.mean()),
        "rg2_std": float(rg2.std()),
        "re2_mean": float(re2.mean()),
        "re2_std": float(re2.std()),
        "n_chains": int(ca_np.shape[0]),
        "N": int(ca_np.shape[1]),
    }
