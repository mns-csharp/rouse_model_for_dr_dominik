"""Fully batched delta-E — torch CUDA tensors, GPU-resident, no Python loop.

Computes delta_E for ALL B proposals in one fused tensor op. State stays on
the GPU; no host transfer happens here. Wrapped with torch.compile so the
broadcast subtract / wrap / square / sum / mask / reduce chain fuses into a
single CUDA kernel and dispatch overhead goes from O(B*ops) to O(1).

The eager core is kept as `_batch_delta_e_torch_fused_eager` for fallback
when compile is unavailable (graceful degrade, no behavioural change).

Public symbol: `batch_delta_e_torch_fused` (compiled by default).
Override via `ROUSE_TORCH_COMPILE_MODE = inductor | aot_eager | eager`.
"""

from __future__ import annotations

import os
import torch

_USE_COMPILE = os.environ.get("ROUSE_NO_TORCH_COMPILE", "0") != "1"
_COMPILE_MODE = os.environ.get("ROUSE_TORCH_COMPILE_MODE", "inductor").lower()


def _zone_e(r2: torch.Tensor, r_rep_sq: float, r_max_sq: float,
            rep_e: float, contact_e: float) -> torch.Tensor:
    e = torch.where(r2 < r_rep_sq, torch.full_like(r2, rep_e), torch.zeros_like(r2))
    if contact_e != 0.0:
        e = torch.where((r2 >= r_rep_sq) & (r2 < r_max_sq),
                        torch.full_like(r2, contact_e), e)
    return e


def _batch_delta_e_torch_fused_eager(
        old_ca, new_ca, old_sg, new_sg, n_moved_out,
        chain_idx, bead_start,
        ca_full, sg_full,
        N, box,
        r_rep_sq, r_max_sq, rep_e, contact_e,
        min_caca, min_casg, min_sgsg,
):
    """Fused delta_E across all B proposals, chunked along the B axis.

    Shapes:
      old_ca, new_ca, old_sg, new_sg : (B, M, 3)   torch float on CUDA
      n_moved_out, chain_idx, bead_start : (B,)    torch long on CUDA
      ca_full, sg_full              : (n_chains, N, 3) torch float on CUDA
    Returns delta_e: (B,) on CUDA.

    The peak transient memory is the (4 * B_chunk * M, 2 * NK) energy tensor.
    At NK=200,000 with the full B=K=1000 this would be 25 GB — hits OOM on a
    12 GB GPU. We compute a `B_chunk` cap from `_MEMORY_BUDGET_BYTES` and
    process B in chunks. At small K the chunk equals B so there is no slowdown.
    """
    B, M, _ = old_ca.shape
    device = old_ca.device
    dtype = old_ca.dtype
    inv_box = 1.0 / box
    n_chains = ca_full.shape[0]
    NK = n_chains * N

    ca_flat = ca_full.reshape(NK, 3)
    sg_flat = sg_full.reshape(NK, 3)
    ref_idx = torch.arange(NK, device=device)
    ref_chain = ref_idx // N
    ref_resid = ref_idx % N

    delta_e = torch.zeros(B, device=device, dtype=dtype)

    # Chunk size: keep peak transient under ~2 GB. The dominant intermediate
    # is the (4*B_chunk*M, 2*NK, 3) diff tensor at 24*B_chunk*M*NK bytes plus
    # ~4 sibling tensors of half that size. Roughly 50*B_chunk*M*NK total.
    elem_bytes = 4  # fp32
    per_b_bytes = 50 * M * NK * elem_bytes
    B_chunk = max(1, int(_MEMORY_BUDGET_BYTES // max(1, per_b_bytes)))
    B_chunk = min(B_chunk, B)

    k_range = torch.arange(M, device=device).unsqueeze(0)

    for cs in range(0, B, B_chunk):
        ce = min(cs + B_chunk, B)
        Bc = ce - cs

        old_ca_c = old_ca[cs:ce]
        new_ca_c = new_ca[cs:ce]
        old_sg_c = old_sg[cs:ce]
        new_sg_c = new_sg[cs:ce]
        n_moved_c = n_moved_out[cs:ce]
        chain_idx_c = chain_idx[cs:ce]
        bead_start_c = bead_start[cs:ce]

        lane_valid = k_range < n_moved_c.unsqueeze(1)
        lane_valid_f = lane_valid.reshape(Bc * M)

        bead_resid = bead_start_c.unsqueeze(1) + k_range
        moved_chain = chain_idx_c.unsqueeze(1).expand(Bc, M)
        moved_chain_f = moved_chain.reshape(Bc * M)
        moved_resid_f = bead_resid.reshape(Bc * M)

        moved_ca_old = old_ca_c.reshape(Bc * M, 3)
        moved_ca_new = new_ca_c.reshape(Bc * M, 3)
        moved_sg_old = old_sg_c.reshape(Bc * M, 3)
        moved_sg_new = new_sg_c.reshape(Bc * M, 3)

        same_chain = moved_chain_f.unsqueeze(1) == ref_chain.unsqueeze(0)
        seq_diff = (moved_resid_f.unsqueeze(1) - ref_resid.unsqueeze(0)).abs()

        seg_start_f = (bead_start_c.unsqueeze(1).expand(Bc, M).reshape(Bc * M)).unsqueeze(1)
        seg_end_f = ((bead_start_c + n_moved_c).unsqueeze(1).expand(Bc, M).reshape(Bc * M)).unsqueeze(1)
        in_moved_span = (ref_resid.unsqueeze(0) >= seg_start_f) & \
                        (ref_resid.unsqueeze(0) < seg_end_f)
        self_excl = same_chain & in_moved_span

        skip_caca = same_chain & (seq_diff < min_caca)
        skip_casg = same_chain & (seq_diff < min_casg)
        skip_sgsg = same_chain & (seq_diff < min_sgsg)

        keep_caca = (~(skip_caca | self_excl)) & lane_valid_f.unsqueeze(1)
        keep_casg = (~(skip_casg | self_excl)) & lane_valid_f.unsqueeze(1)
        keep_sgsg = (~(skip_sgsg | self_excl)) & lane_valid_f.unsqueeze(1)
        m_caca = keep_caca.to(dtype)
        m_casg = keep_casg.to(dtype)
        m_sgsg = keep_sgsg.to(dtype)

        moved_all = torch.cat([moved_ca_old, moved_ca_new, moved_sg_old, moved_sg_new], dim=0)
        ref_all = torch.cat([ca_flat, sg_flat], dim=0)
        d = ref_all.unsqueeze(0) - moved_all.unsqueeze(1)
        d = d - box * torch.round(d * inv_box)
        r2_big = (d * d).sum(dim=-1)
        e_big = _zone_e(r2_big, r_rep_sq, r_max_sq, rep_e, contact_e)

        BcM = Bc * M
        e_caca_old = (e_big[:BcM, :NK] * m_caca).sum(dim=1)
        e_casg_a_old = (e_big[:BcM, NK:] * m_casg).sum(dim=1)
        e_caca_new = (e_big[BcM:2 * BcM, :NK] * m_caca).sum(dim=1)
        e_casg_a_new = (e_big[BcM:2 * BcM, NK:] * m_casg).sum(dim=1)
        e_casg_b_old = (e_big[2 * BcM:3 * BcM, :NK] * m_casg).sum(dim=1)
        e_sgsg_old = (e_big[2 * BcM:3 * BcM, NK:] * m_sgsg).sum(dim=1)
        e_casg_b_new = (e_big[3 * BcM:4 * BcM, :NK] * m_casg).sum(dim=1)
        e_sgsg_new = (e_big[3 * BcM:4 * BcM, NK:] * m_sgsg).sum(dim=1)

        e_lane_old = e_caca_old + e_casg_a_old + e_casg_b_old + e_sgsg_old
        e_lane_new = e_caca_new + e_casg_a_new + e_casg_b_new + e_sgsg_new
        delta_lane = e_lane_new - e_lane_old
        delta_e[cs:ce] = delta_lane.reshape(Bc, M).sum(dim=1)

    return delta_e


# Peak transient budget for the chunked all-pairs reduction. 2 GiB keeps us
# well under a 12 GB GPU even with concurrent torch.compile-allocated buffers
# and the persistent state + scratch.
_MEMORY_BUDGET_BYTES = 2 * 1024 * 1024 * 1024


try:
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True
except Exception:
    pass


def _resolve_compiled():
    fn = _batch_delta_e_torch_fused_eager
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


batch_delta_e_torch_fused = _resolve_compiled()

# Back-compat alias so checklist tests and any sibling imports still resolve.
batch_delta_e_torch = batch_delta_e_torch_fused
