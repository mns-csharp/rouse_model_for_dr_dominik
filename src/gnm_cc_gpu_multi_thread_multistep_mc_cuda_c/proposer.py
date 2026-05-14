"""Segment-batch proposer -- real CUDA C via cp.RawKernel.

Single-segment (B=1) proposer optimized for the conventional MC inner
loop. All persistent tensors live in a `GpuScratch` instance whose
cupy dlpack views are cached once at sim init, so per-call dlpack
interop drops from ~15 calls/segment to zero.
"""

from __future__ import annotations

import numpy as np
import torch


SEG_N_TERMINAL = 0
SEG_C_TERMINAL = 1
SEG_INNER = 2
SEG_BOTH = 3

MTYPE_HINGE = 0
MTYPE_N_TAIL = 1
MTYPE_C_TAIL = 2


def _torch_to_cupy(t: torch.Tensor):
    """Legacy fallback used by callers that haven't migrated to scratch yet."""
    import cupy as cp
    if hasattr(cp, "from_dlpack"):
        try:
            return cp.from_dlpack(t)
        except (TypeError, ValueError):
            pass
    from torch.utils.dlpack import to_dlpack
    return cp.fromDlpack(to_dlpack(t))


def propose_batch_torch(
        ca_t: torch.Tensor, sg_t: torch.Tensor,
        meta_t: torch.Tensor, rand_t: torch.Tensor,
        max_moved: int,
        N: int, box: float, max_angle: float,
):
    """Legacy free-buffer dispatch used by synthetic-checklist tests.

    The hot path uses propose_segment_b1 with a pre-allocated GpuScratch.
    """
    import cupy as cp
    from . import cuda_kernels as ck

    device = ca_t.device
    dtype = ca_t.dtype
    B = int(meta_t.shape[0])
    M = max(int(max_moved), 1)

    if dtype != torch.float32:
        raise TypeError(
            "cuda_c propose_batch_kernel expects float32 chain state; "
            f"got {dtype}."
        )

    old_ca = torch.zeros(B, M, 3, dtype=dtype, device=device)
    new_ca = torch.zeros(B, M, 3, dtype=dtype, device=device)
    old_sg = torch.zeros(B, M, 3, dtype=dtype, device=device)
    new_sg = torch.zeros(B, M, 3, dtype=dtype, device=device)
    n_moved_out = torch.zeros(B, dtype=torch.int64, device=device)
    move_type = torch.zeros(B, dtype=torch.int64, device=device)

    chain_idx_t = meta_t[:, 0].contiguous().to(torch.int64)
    bead_start_t = meta_t[:, 1].contiguous().to(torch.int64)
    n_moved_in_t = meta_t[:, 2].contiguous().to(torch.int64)
    seg_type_t = meta_t[:, 3].contiguous().to(torch.int64)
    anchor_a_t = meta_t[:, 4].contiguous().to(torch.int64)
    anchor_b_t = meta_t[:, 5].contiguous().to(torch.int64)
    rand_t = rand_t.contiguous()
    if rand_t.dtype != torch.float32:
        rand_t = rand_t.to(torch.float32)

    ca_flat = ca_t.contiguous().view(-1, 3)
    sg_flat = sg_t.contiguous().view(-1, 3)

    cp_ca = _torch_to_cupy(ca_flat)
    cp_sg = _torch_to_cupy(sg_flat)
    cp_old_ca = _torch_to_cupy(old_ca)
    cp_new_ca = _torch_to_cupy(new_ca)
    cp_old_sg = _torch_to_cupy(old_sg)
    cp_new_sg = _torch_to_cupy(new_sg)
    cp_ch = _torch_to_cupy(chain_idx_t)
    cp_bs = _torch_to_cupy(bead_start_t)
    cp_nm = _torch_to_cupy(n_moved_in_t)
    cp_st = _torch_to_cupy(seg_type_t)
    cp_aa = _torch_to_cupy(anchor_a_t)
    cp_ab = _torch_to_cupy(anchor_b_t)
    cp_ra = _torch_to_cupy(rand_t)
    cp_no = _torch_to_cupy(n_moved_out)
    cp_mt = _torch_to_cupy(move_type)

    inv_box = 1.0 / box
    half_box = 0.5 * box
    block = (32, 1, 1)
    grid = (B, 1, 1)
    smem_bytes = (M * 3 + 8) * 4

    kernel = ck.get_propose_kernel()
    torch_stream = torch.cuda.current_stream(device)
    cp_stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)
    with cp_stream:
        kernel(grid, block, (
            cp_ca, cp_sg,
            cp_old_ca, cp_new_ca, cp_old_sg, cp_new_sg,
            cp_ch, cp_bs, cp_nm, cp_st, cp_aa, cp_ab,
            cp_ra,
            cp_no, cp_mt,
            np.int32(N), np.int32(M), np.int32(B),
            np.float32(box), np.float32(inv_box), np.float32(half_box),
            np.float32(max_angle), np.float32(0.0),
        ), shared_mem=smem_bytes)

    return old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type


def propose_segment_b1(scratch, ca_t: torch.Tensor, sg_t: torch.Tensor,
                       gs: int, N: int, box: float, max_angle: float):
    """Launch propose_batch_kernel for one segment (B=1) using cached views.

    Returns the scratch buffers (no allocation): old_ca, new_ca, old_sg,
    new_sg, n_moved_out, move_type. All [1, M, 3] or [1].
    """
    import cupy as cp
    from . import cuda_kernels as ck

    if ca_t.dtype != torch.float32:
        raise TypeError(
            "cuda_c propose_batch_kernel expects float32 chain state; "
            f"got {ca_t.dtype}."
        )

    cp_ca = scratch.cp_state_ca(ca_t)
    cp_sg = scratch.cp_state_sg(sg_t)
    cp_old_ca = scratch.cp("old_ca")
    cp_new_ca = scratch.cp("new_ca")
    cp_old_sg = scratch.cp("old_sg")
    cp_new_sg = scratch.cp("new_sg")
    cp_no = scratch.cp("n_moved_out")
    cp_mt = scratch.cp("move_type")

    # 1-element slices off pre-cached column views (no dlpack hits).
    cp_ch = scratch.cp("col_chain")[gs:gs + 1]
    cp_bs = scratch.cp("col_bead")[gs:gs + 1]
    cp_nm = scratch.cp("col_nmov")[gs:gs + 1]
    cp_st = scratch.cp("col_seg")[gs:gs + 1]
    cp_aa = scratch.cp("col_aa")[gs:gs + 1]
    cp_ab = scratch.cp("col_ab")[gs:gs + 1]
    cp_ra = scratch.cp("rand_buf")[gs:gs + 1]

    M = scratch.M
    B = 1
    inv_box = 1.0 / box
    half_box = 0.5 * box
    block = (32, 1, 1)
    grid = (B, 1, 1)
    smem_bytes = (M * 3 + 8) * 4

    kernel = ck.get_propose_kernel()
    torch_stream = torch.cuda.current_stream(ca_t.device)
    cp_stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)
    with cp_stream:
        kernel(grid, block, (
            cp_ca, cp_sg,
            cp_old_ca, cp_new_ca, cp_old_sg, cp_new_sg,
            cp_ch, cp_bs, cp_nm, cp_st, cp_aa, cp_ab,
            cp_ra,
            cp_no, cp_mt,
            np.int32(N), np.int32(M), np.int32(B),
            np.float32(box), np.float32(inv_box), np.float32(half_box),
            np.float32(max_angle), np.float32(0.0),
        ), shared_mem=smem_bytes)

    return (scratch.old_ca, scratch.new_ca, scratch.old_sg, scratch.new_sg,
            scratch.n_moved_out, scratch.move_type)
