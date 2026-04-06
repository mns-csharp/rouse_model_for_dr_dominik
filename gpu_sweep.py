"""
GPU sweep orchestrator: runs the full MC sweep on GPU via CuPy RawKernels.

Analogous to fast_sweep.py but positions stay on GPU. CPU only handles
the batch permutation logic and sequential Metropolis acceptance loop.

Architecture per batch:
  GPU: segment_proposal_kernel → delta_e_kernel → emm_correction_kernel
  D2H: delta_e[B] + correction[B,B]  (~3KB)
  CPU: sequential Metropolis with rank-1 corrections
  GPU: apply_moves_kernel (scatter accepted moves)
"""

import math
import numpy as np
import torch
import cupy as cp

from .chain import SegmentInfo
from .config import SimulationConfig
from .gpu_cell_list import GPUCellList
from .gpu_kernels import GPUKernelManager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOVE_SIZE = 20        # Batch size for multistep MC
REBUILD_INTERVAL = 3  # Rebuild cell list every N batches


def _torch_to_cupy(t: torch.Tensor) -> cp.ndarray:
    """Zero-copy convert PyTorch CUDA tensor to CuPy array via DLPack."""
    return cp.from_dlpack(t)


def _cupy_to_torch(a: cp.ndarray, device: torch.device) -> torch.Tensor:
    """Zero-copy convert CuPy array to PyTorch CUDA tensor via DLPack."""
    return torch.from_dlpack(a)


# ---------------------------------------------------------------------------
# Metropolis acceptance (CPU, athermal)
# ---------------------------------------------------------------------------

def _metropolis_accept(delta_e: float, kBT: float, rng: np.random.RandomState) -> bool:
    """Metropolis acceptance with overflow guards."""
    if delta_e <= 0.0:
        return True
    exponent = -delta_e / kBT
    if exponent <= -745.0:
        return False
    if exponent >= 709.0:
        return True
    return rng.random() < math.exp(exponent)


# ---------------------------------------------------------------------------
# GPU sweep class
# ---------------------------------------------------------------------------

class GPUFastSweep:
    """
    Manages GPU state and kernels for the MC sweep.

    Holds compiled CuPy kernels, GPU cell list, and pre-allocated
    work buffers. Reused across sweeps.
    """

    def __init__(self, cfg: SimulationConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.km = GPUKernelManager()
        self.km.compile_all()

        # GPU cell list
        self.cell_list = GPUCellList(cfg.box_size, cfg.r_max, device)

        # Pre-allocated FP32 position buffer (updated from FP64 state)
        n_total = cfg.n_chains * cfg.N
        self.pos_f32 = torch.empty(n_total, 3, dtype=torch.float32, device=device)

        # Physical params for kernels
        self.box = cfg.box_size
        self.inv_box = 1.0 / cfg.box_size
        self.half_box = cfg.box_size / 2.0
        self.r_rep_sq = cfg.r_rep_sq
        self.rep_e = cfg.repulsive_energy
        self.kBT = cfg.kBT

        # Pre-allocated scratch buffers for delta_e kernel
        # Bitmap: [max_B, nc3] bytes for visited-cell tracking
        # Visited list: [max_B, max_visited] ints
        nc3 = self.cell_list._nc3
        max_B = max(MOVE_SIZE, cfg.n_chains)  # largest batch (pivots can be n_chains)
        self._max_visited = min(nc3, 27 * 2 * cfg.N)  # conservative upper bound
        self._visited_bitmaps = torch.zeros(
            max_B, nc3, dtype=torch.uint8, device=device)
        self._visited_lists = torch.zeros(
            max_B, self._max_visited, dtype=torch.int32, device=device)
        self._n_visited_out = torch.zeros(
            max_B, dtype=torch.int32, device=device)

    def _sync_f32(self, positions: torch.Tensor):
        """Copy FP64 positions to FP32 buffer for kernel input."""
        self.pos_f32.copy_(positions.view(-1, 3).float())

    def _rebuild_cell_list(self):
        """Rebuild GPU cell list from current FP32 positions."""
        self.cell_list.build(self.pos_f32)

    def _prepare_segment_metadata(self, seg_list, seg_info: SegmentInfo, N: int):
        """
        Prepare GPU arrays for segment proposal kernel.

        Args:
            seg_list: list of (chain_idx, local_seg) tuples for this batch
            seg_info: SegmentInfo
            N: beads per chain

        Returns:
            dict of GPU int32 tensors + max_moved + move_types list
        """
        B = len(seg_list)
        chain_idx = np.empty(B, dtype=np.int32)
        bead_start = np.empty(B, dtype=np.int32)
        n_moved = np.empty(B, dtype=np.int32)
        seg_type_arr = np.empty(B, dtype=np.int32)
        anchor_a = np.empty(B, dtype=np.int32)
        anchor_b = np.empty(B, dtype=np.int32)
        move_types = []

        max_moved = 0
        for i, (ci, ls) in enumerate(seg_list):
            stype = seg_info.get_segment_type(ls)
            seg_start, seg_end = seg_info.get_segment_range(ci, ls)

            if stype == SegmentInfo.INNER:
                # Hinge: move seg_start..seg_end
                ms, me = seg_start, seg_end
                a = max(seg_start - 1, 0)
                b = min(seg_end, N - 1)
                move_types.append('hinge')
            elif stype == SegmentInfo.N_TERMINAL or stype == SegmentInfo.BOTH:
                # N-terminal: move 0..seg_end
                ms, me = 0, seg_end
                a = min(seg_end, N - 1)
                b = min(seg_end + 1, N - 1)
                move_types.append('n_tail')
            elif stype == SegmentInfo.C_TERMINAL:
                # C-terminal: move seg_start..N
                ms, me = seg_start, N
                a = max(seg_start - 1, 0)
                b = max(seg_start - 2, 0)
                move_types.append('c_tail')
            else:
                raise ValueError(f"Unknown segment type: {stype}")

            chain_idx[i] = ci
            bead_start[i] = ms
            n_moved[i] = me - ms
            seg_type_arr[i] = stype
            # Anchor indices are global (chain_idx * N + local_bead)
            anchor_a[i] = ci * N + a
            anchor_b[i] = ci * N + b
            max_moved = max(max_moved, me - ms)

        dev = self.device
        return {
            'chain_idx': torch.from_numpy(chain_idx).to(dev),
            'bead_start': torch.from_numpy(bead_start).to(dev),
            'n_moved': torch.from_numpy(n_moved).to(dev),
            'seg_type': torch.from_numpy(seg_type_arr).to(dev),
            'anchor_a': torch.from_numpy(anchor_a).to(dev),
            'anchor_b': torch.from_numpy(anchor_b).to(dev),
            'max_moved': max_moved,
            'move_types': move_types,
            'B': B,
            # Keep numpy copies for CPU acceptance loop
            'chain_idx_np': chain_idx,
            'bead_start_np': bead_start,
            'n_moved_np': n_moved,
        }

    def _launch_segment_proposals(self, meta, rng: np.random.RandomState):
        """Launch segment proposal kernel, return old_pos and new_pos GPU tensors."""
        B = meta['B']
        mm = meta['max_moved']
        N = self.cfg.N
        dev = self.device

        # Pre-generate random angles on GPU
        # angle = (2*rand - 1) * max_angle
        rand_raw = torch.rand(B, dtype=torch.float32, device=dev)
        rand_angles = (2.0 * rand_raw - 1.0) * self.cfg.max_angle_hinge

        # Allocate output buffers
        old_pos = torch.zeros(B, mm, 3, dtype=torch.float32, device=dev)
        new_pos = torch.zeros(B, mm, 3, dtype=torch.float32, device=dev)

        # Launch kernel
        kernel = self.km['segment_proposal']
        kernel(
            (B,), (1,),
            (_torch_to_cupy(self.pos_f32),
             _torch_to_cupy(old_pos),
             _torch_to_cupy(new_pos),
             _torch_to_cupy(meta['chain_idx']),
             _torch_to_cupy(meta['bead_start']),
             _torch_to_cupy(meta['n_moved']),
             _torch_to_cupy(meta['seg_type']),
             _torch_to_cupy(meta['anchor_a']),
             _torch_to_cupy(meta['anchor_b']),
             _torch_to_cupy(rand_angles),
             np.float32(self.box), np.float32(self.inv_box),
             np.float32(self.half_box),
             np.int32(N), np.int32(mm), np.int32(B))
        )

        return old_pos, new_pos

    def _launch_delta_e(self, old_pos, new_pos, meta):
        """Launch delta-E kernel, return delta_e GPU tensor."""
        B = meta['B']
        mm = meta['max_moved']
        N = self.cfg.N
        dev = self.device
        cl = self.cell_list

        # Compute global starts/ends
        global_starts = (meta['chain_idx'] * N + meta['bead_start']).int()
        global_ends = (global_starts + meta['n_moved']).int()

        delta_e_out = torch.zeros(B, dtype=torch.float32, device=dev)

        max_nm = int(meta['n_moved'].max().item()) if B > 0 else 0
        if max_nm == 0:
            return delta_e_out, global_starts

        # Shared memory: old_pos[nm][3] + new_pos[nm][3]
        smem_bytes = max_nm * 6 * 4 + 256  # positions + padding

        kernel = self.km['delta_e']
        kernel(
            (B,), (256,),
            (_torch_to_cupy(self.pos_f32.view(-1)),
             _torch_to_cupy(old_pos),
             _torch_to_cupy(new_pos),
             _torch_to_cupy(meta['n_moved']),
             _torch_to_cupy(global_starts),
             _torch_to_cupy(global_ends),
             _torch_to_cupy(delta_e_out),
             _torch_to_cupy(cl.sorted_order),
             _torch_to_cupy(cl.cell_starts),
             _torch_to_cupy(cl.cell_counts),
             _torch_to_cupy(cl.neighbor_offsets),
             _torch_to_cupy(self._visited_bitmaps[:B]),
             _torch_to_cupy(self._visited_lists[:B]),
             _torch_to_cupy(self._n_visited_out[:B]),
             np.float32(self.box), np.float32(self.inv_box),
             np.float32(self.half_box),
             np.float32(cl._inv_cs), np.int32(cl._nc), np.int32(cl._nc3),
             np.float32(self.r_rep_sq), np.float32(self.rep_e),
             np.int32(mm), np.int32(B), np.int32(self._max_visited)),
            shared_mem=smem_bytes
        )

        return delta_e_out, global_starts

    def _launch_emm(self, old_pos, new_pos, meta):
        """Launch fused EMM correction kernel, return single [B, B] GPU tensor.

        Optimizations:
          - Bounding-sphere filtering: skip distant pairs that cannot interact
          - Fused kernel: computes correction = (E11-E01)-(E10-E00) directly
          - Single output matrix instead of four (4× less memory + D2H transfer)
        """
        B = meta['B']
        mm = meta['max_moved']
        dev = self.device
        n_moved_np = meta['n_moved_np']

        correction = torch.zeros(B, B, dtype=torch.float32, device=dev)

        # Find valid proposals
        valid = [i for i in range(B) if n_moved_np[i] > 0]
        if len(valid) < 2:
            return correction

        # Bounding-sphere filtering: compute centroid + radius for each proposal
        box = self.box
        inv_box = self.inv_box
        r_rep = math.sqrt(self.r_rep_sq)

        centroids = np.zeros((B, 3), dtype=np.float32)
        radii = np.zeros(B, dtype=np.float32)

        # Pull positions to CPU for bounding sphere computation (small data)
        old_cpu = old_pos.cpu().numpy()  # [B, mm, 3]
        new_cpu = new_pos.cpu().numpy()

        for i in valid:
            nm = int(n_moved_np[i])
            all_pos = np.concatenate([old_cpu[i, :nm], new_cpu[i, :nm]], axis=0)
            centroid = all_pos.mean(axis=0)
            centroids[i] = centroid
            d = all_pos - centroid
            d = d - box * np.round(d * inv_box)
            radii[i] = np.sqrt((d * d).sum(axis=1).max())

        # Build sparse pair list: only pairs whose bounding spheres overlap
        pairs_i = []
        pairs_j = []
        for ii in range(len(valid)):
            i = valid[ii]
            for jj in range(ii + 1, len(valid)):
                j = valid[jj]
                dc = centroids[i] - centroids[j]
                dc = dc - box * np.round(dc * inv_box)
                dist = np.sqrt((dc * dc).sum())
                if dist < radii[i] + radii[j] + r_rep:
                    pairs_i.append(i)
                    pairs_j.append(j)

        n_pairs = len(pairs_i)
        if n_pairs == 0:
            return correction

        pair_i_gpu = torch.tensor(pairs_i, dtype=torch.int32, device=dev)
        pair_j_gpu = torch.tensor(pairs_j, dtype=torch.int32, device=dev)

        kernel = self.km['emm_correction']
        kernel(
            (n_pairs,), (256,),
            (_torch_to_cupy(old_pos),
             _torch_to_cupy(new_pos),
             _torch_to_cupy(meta['n_moved']),
             _torch_to_cupy(correction),
             _torch_to_cupy(pair_i_gpu),
             _torch_to_cupy(pair_j_gpu),
             np.float32(self.box), np.float32(self.inv_box),
             np.float32(self.r_rep_sq), np.float32(self.rep_e),
             np.int32(mm), np.int32(B), np.int32(n_pairs))
        )

        return correction

    def _apply_accepted_moves(self, positions: torch.Tensor,
                              new_pos: torch.Tensor,
                              accepted_indices: list,
                              global_starts: torch.Tensor,
                              meta: dict):
        """Scatter accepted proposals into FP64 position state."""
        if not accepted_indices:
            return

        n_accepted = len(accepted_indices)
        dev = self.device
        mm = meta['max_moved']

        accepted_gpu = torch.tensor(accepted_indices, dtype=torch.int32, device=dev)

        kernel = self.km['apply_moves']
        pos_flat = positions.view(-1, 3).contiguous()
        kernel(
            (n_accepted,), (256,),
            (_torch_to_cupy(pos_flat),
             _torch_to_cupy(new_pos),
             _torch_to_cupy(accepted_gpu),
             _torch_to_cupy(global_starts),
             _torch_to_cupy(meta['n_moved']),
             np.int32(mm), np.int32(n_accepted))
        )
        # Copy back if view was not contiguous (rare, but safe)
        if not positions.view(-1, 3).is_contiguous():
            positions.view(-1, 3).copy_(pos_flat)

        # Also update FP32 cache for subsequent batches
        for idx in accepted_indices:
            ci = meta['chain_idx_np'][idx]
            bs = meta['bead_start_np'][idx]
            nm = meta['n_moved_np'][idx]
            gs = ci * self.cfg.N + bs
            # Copy from FP64 state to FP32 cache
            self.pos_f32[gs:gs+nm] = positions.view(-1, 3)[gs:gs+nm].float()


# ---------------------------------------------------------------------------
# Full sweep function
# ---------------------------------------------------------------------------

def gpu_perform_sweep(positions: torch.Tensor, seg_info: SegmentInfo,
                      sweep: GPUFastSweep, cfg: SimulationConfig,
                      stats, rng: np.random.RandomState,
                      skip_pivot: bool = False):
    """
    Perform one full MC sweep on GPU.

    Args:
        positions: [n_chains, N, 3] FP64 CUDA tensor (modified in-place)
        seg_info: SegmentInfo
        sweep: GPUFastSweep (holds kernels, cell list, buffers)
        cfg: SimulationConfig
        stats: SimulationStats
        rng: numpy RandomState for CPU acceptance decisions
        skip_pivot: if True, skip Phase 2 pivot moves

    Phase 1: Batched multistep segment moves with rank-1 corrections
    Phase 2: Sequential pivot moves (batched on GPU)
    """
    N = cfg.N
    n_chains = cfg.n_chains
    total_segs = seg_info.total_segments
    kBT = cfg.kBT
    dev = sweep.device

    # Update FP32 position cache from FP64 state
    sweep._sync_f32(positions)

    # ── Build batched permutation with same-chain exclusion ──────────
    # (identical logic to fast_sweep.py)
    batch_size = MOVE_SIZE
    segs_per_chain = seg_info.segs_per_chain

    buckets = [[] for _ in range(n_chains)]
    for gs in range(total_segs):
        buckets[seg_info.seg_chain[gs]].append(gs)
    for bucket in buckets:
        rng.shuffle(bucket)

    rounds = []
    for k in range(segs_per_chain):
        round_segs = [bucket[k] for bucket in buckets if k < len(bucket)]
        rng.shuffle(round_segs)
        rounds.append(round_segs)

    perm = []
    round_boundaries = []
    for round_segs in rounds:
        round_boundaries.append(len(perm))
        perm.extend(round_segs)
    round_boundaries.append(len(perm))

    batch_ranges = []
    for ri in range(len(rounds)):
        r_start = round_boundaries[ri]
        r_end = round_boundaries[ri + 1]
        for bs in range(r_start, r_end, batch_size):
            be = min(bs + batch_size, r_end)
            batch_ranges.append((bs, be))

    # ── Phase 1: Segment moves ──────────────────────────────────────
    for b_idx, (b_start, b_end) in enumerate(batch_ranges):

        # Rebuild cell list periodically
        if b_idx % REBUILD_INTERVAL == 0:
            sweep._rebuild_cell_list()

        # Build segment list for this batch
        seg_list = []
        for idx in range(b_start, b_end):
            gs = perm[idx]
            seg_list.append((seg_info.seg_chain[gs], seg_info.seg_local[gs]))

        # Prepare GPU metadata
        meta = sweep._prepare_segment_metadata(seg_list, seg_info, N)
        B = meta['B']

        if B == 0:
            continue

        # GPU: propose all moves
        old_pos, new_pos = sweep._launch_segment_proposals(meta, rng)

        # GPU: compute delta-E
        delta_e_gpu, global_starts = sweep._launch_delta_e(old_pos, new_pos, meta)

        # GPU: compute fused correction matrix
        correction_gpu = sweep._launch_emm(old_pos, new_pos, meta)

        # D2H: copy results to CPU (1 matrix instead of 4)
        delta_e = delta_e_gpu.cpu().tolist()
        corr = correction_gpu.cpu().tolist()

        # CPU: sequential Metropolis acceptance with rank-1 corrections
        n_moved_np = meta['n_moved_np']
        move_types = meta['move_types']
        accepted_indices = []

        for i in range(B):
            if n_moved_np[i] == 0:
                continue

            accepted = _metropolis_accept(delta_e[i], kBT, rng)
            stats.record(move_types[i], accepted)

            if not accepted:
                continue

            accepted_indices.append(i)

            # Rank-1 energy correction from fused correction matrix
            for j in range(i + 1, B):
                if n_moved_np[j] == 0:
                    continue
                c = corr[i][j]
                if c != 0.0:
                    delta_e[j] += c

        # GPU: apply accepted moves
        sweep._apply_accepted_moves(positions, new_pos, accepted_indices,
                                    global_starts, meta)

    # ── Phase 2: Pivot moves ────────────────────────────────────────
    if skip_pivot:
        return

    # Rebuild cell list before pivots
    sweep._sync_f32(positions)
    sweep._rebuild_cell_list()

    # Batch pivots like segments for consistency
    chain_perm = rng.permutation(n_chains).tolist()

    # Process pivots in batches
    pivot_batch_size = MOVE_SIZE
    for pb_start in range(0, n_chains, pivot_batch_size):
        pb_end = min(pb_start + pivot_batch_size, n_chains)
        batch_chains = chain_perm[pb_start:pb_end]
        Bp = len(batch_chains)

        # Max possible moved beads for a pivot is N-1
        max_pivot_moved = N - 1 if N > 2 else N

        # Pre-generate random numbers for pivot proposals
        rand_buf = torch.rand(Bp, 6, dtype=torch.float32, device=dev)

        # Allocate output
        old_pos_pivot = torch.zeros(Bp, max_pivot_moved, 3,
                                    dtype=torch.float32, device=dev)
        new_pos_pivot = torch.zeros(Bp, max_pivot_moved, 3,
                                    dtype=torch.float32, device=dev)
        rot_start_out = torch.zeros(Bp, dtype=torch.int32, device=dev)
        n_moved_out = torch.zeros(Bp, dtype=torch.int32, device=dev)
        chain_perm_gpu = torch.tensor(batch_chains, dtype=torch.int32, device=dev)

        # Launch pivot proposal kernel
        kernel = sweep.km['pivot_proposal']
        kernel(
            (Bp,), (1,),
            (_torch_to_cupy(sweep.pos_f32),
             _torch_to_cupy(old_pos_pivot),
             _torch_to_cupy(new_pos_pivot),
             _torch_to_cupy(rot_start_out),
             _torch_to_cupy(n_moved_out),
             _torch_to_cupy(rand_buf),
             _torch_to_cupy(chain_perm_gpu),
             np.float32(sweep.box), np.float32(sweep.inv_box),
             np.float32(sweep.half_box),
             np.int32(N), np.int32(n_chains), np.int32(max_pivot_moved))
        )

        # Read back pivot metadata
        rot_starts = rot_start_out.cpu().numpy()
        n_moveds = n_moved_out.cpu().numpy()

        # Build meta dict for delta-E and Emm
        pivot_meta = {
            'chain_idx': chain_perm_gpu,
            'bead_start': rot_start_out,
            'n_moved': n_moved_out,
            'max_moved': max_pivot_moved,
            'B': Bp,
            'chain_idx_np': np.array(batch_chains, dtype=np.int32),
            'bead_start_np': rot_starts.astype(np.int32),
            'n_moved_np': n_moveds.astype(np.int32),
        }

        # GPU: delta-E (reuse the standard launch method)
        delta_e_pivot_gpu, global_starts_pivot = sweep._launch_delta_e(
            old_pos_pivot, new_pos_pivot, pivot_meta)

        # GPU: fused correction matrix for rank-1
        correction_gpu = sweep._launch_emm(
            old_pos_pivot, new_pos_pivot, pivot_meta)

        # D2H (1 matrix instead of 4)
        delta_e = delta_e_pivot_gpu.cpu().tolist()
        corr = correction_gpu.cpu().tolist()

        # CPU: sequential Metropolis with rank-1
        accepted_indices = []
        for i in range(Bp):
            if n_moveds[i] == 0:
                continue

            accepted = _metropolis_accept(delta_e[i], kBT, rng)
            stats.record('pivot', accepted)

            if not accepted:
                continue

            accepted_indices.append(i)

            for j in range(i + 1, Bp):
                if n_moveds[j] == 0:
                    continue
                c = corr[i][j]
                if c != 0.0:
                    delta_e[j] += c

        # GPU: apply accepted pivot moves
        if accepted_indices:
            n_acc = len(accepted_indices)
            acc_gpu = torch.tensor(accepted_indices, dtype=torch.int32, device=dev)

            apply_kernel = sweep.km['apply_moves']
            apply_kernel(
                (n_acc,), (256,),
                (_torch_to_cupy(positions.view(-1, 3)),
                 _torch_to_cupy(new_pos_pivot),
                 _torch_to_cupy(acc_gpu),
                 _torch_to_cupy(global_starts_pivot),
                 _torch_to_cupy(n_moved_out),
                 np.int32(max_pivot_moved), np.int32(n_acc))
            )

            # Update FP32 cache
            for idx in accepted_indices:
                ci = batch_chains[idx]
                bs = int(rot_starts[idx])
                nm = int(n_moveds[idx])
                gs = ci * N + bs
                sweep.pos_f32[gs:gs+nm] = positions.view(-1, 3)[gs:gs+nm].float()
