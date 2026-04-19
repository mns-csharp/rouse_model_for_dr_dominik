"""GPUFastSweep — MC sweep orchestrator running on GPU via CuPy RawKernels.

Holds compiled CuPy kernels, the GPU cell list, and pre-allocated scratch
buffers that survive across sweeps. The `perform_sweep` classmethod drives
one full sweep; `perform_pivot_phase` runs just the pivot phase.

CuPy is imported lazily inside `__init__` so the module imports cleanly
on CuPy-less hosts; construction raises ImportError there.
"""

import math

import numpy as np
import torch

from rouse_model_python.src.libs.chain.segment_info import SegmentInfo
from rouse_model_python.src.libs.config.simulation_config import SimulationConfig
from rouse_model_python.src.libs.paths.gpu.gpu_cell_list import GPUCellList
from rouse_model_python.src.libs.paths.gpu.gpu_kernel_manager import GPUKernelManager


class GPUFastSweep:
    MOVE_SIZE = 20
    REBUILD_INTERVAL = 3
    NVTX_ENABLED = False

    @staticmethod
    def _nvtx_push(name: str) -> None:
        if GPUFastSweep.NVTX_ENABLED:
            torch.cuda.nvtx.range_push(name)

    @staticmethod
    def _nvtx_pop() -> None:
        if GPUFastSweep.NVTX_ENABLED:
            torch.cuda.nvtx.range_pop()

    def __init__(self, cfg: SimulationConfig, device: torch.device):
        import cupy as cp
        self._cp = cp
        self.cfg = cfg
        self.device = device
        self.km = GPUKernelManager()
        self.km.compile_all()

        self.cell_list = GPUCellList(cfg.box_size, cfg.r_max, device)

        n_total = cfg.n_chains * cfg.N
        self.pos_f32 = torch.empty(n_total, 3, dtype=torch.float32, device=device)

        self.box = cfg.box_size
        self.inv_box = 1.0 / cfg.box_size
        self.half_box = cfg.box_size / 2.0
        self.r_rep_sq = cfg.r_rep_sq
        self.rep_e = cfg.repulsive_energy
        self.kBT = cfg.kBT

        nc3 = self.cell_list._nc3
        max_B = max(GPUFastSweep.MOVE_SIZE, cfg.n_chains)
        self._max_visited = min(nc3, 27 * 2 * cfg.N)
        self._visited_bitmaps = torch.zeros(
            max_B, nc3, dtype=torch.uint8, device=device)
        self._visited_lists = torch.zeros(
            max_B, self._max_visited, dtype=torch.int32, device=device)
        self._n_visited_out = torch.zeros(
            max_B, dtype=torch.int32, device=device)

    def _to_cupy(self, t: torch.Tensor):
        return self._cp.from_dlpack(t)

    @staticmethod
    def metropolis_accept(delta_e: float, kBT: float, rng) -> bool:
        if delta_e <= 0.0:
            return True
        exponent = -delta_e / kBT
        if exponent <= -745.0:
            return False
        if exponent >= 709.0:
            return True
        return rng.random() < math.exp(exponent)

    def _sync_f32(self, positions: torch.Tensor) -> None:
        self.pos_f32.copy_(positions.view(-1, 3).float())

    def _rebuild_cell_list(self) -> None:
        self.cell_list.build(self.pos_f32)

    def _prepare_segment_metadata(self, seg_list, seg_info: SegmentInfo, N: int) -> dict:
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
                ms, me = seg_start, seg_end
                a = max(seg_start - 1, 0)
                b = min(seg_end, N - 1)
                move_types.append('hinge')
            elif stype == SegmentInfo.N_TERMINAL or stype == SegmentInfo.BOTH:
                ms, me = 0, seg_end
                a = min(seg_end, N - 1)
                b = min(seg_end + 1, N - 1)
                move_types.append('n_tail')
            elif stype == SegmentInfo.C_TERMINAL:
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
            'chain_idx_np': chain_idx,
            'bead_start_np': bead_start,
            'n_moved_np': n_moved,
        }

    def _launch_segment_proposals(self, meta, rng):
        B = meta['B']
        mm = meta['max_moved']
        N = self.cfg.N
        dev = self.device

        rand_raw = torch.rand(B, dtype=torch.float32, device=dev)
        rand_angles = (2.0 * rand_raw - 1.0) * self.cfg.max_angle_hinge

        old_pos = torch.zeros(B, mm, 3, dtype=torch.float32, device=dev)
        new_pos = torch.zeros(B, mm, 3, dtype=torch.float32, device=dev)

        kernel = self.km['segment_proposal']
        max_disp = self.cfg.l0 * (1.0 + 0.05)
        kernel(
            (B,), (1,),
            (self._to_cupy(self.pos_f32),
             self._to_cupy(old_pos),
             self._to_cupy(new_pos),
             self._to_cupy(meta['chain_idx']),
             self._to_cupy(meta['bead_start']),
             self._to_cupy(meta['n_moved']),
             self._to_cupy(meta['seg_type']),
             self._to_cupy(meta['anchor_a']),
             self._to_cupy(meta['anchor_b']),
             self._to_cupy(rand_angles),
             np.float32(self.box), np.float32(self.inv_box),
             np.float32(self.half_box),
             np.float32(max_disp),
             np.int32(N), np.int32(mm), np.int32(B))
        )

        return old_pos, new_pos

    def _launch_delta_e(self, old_pos, new_pos, meta):
        B = meta['B']
        mm = meta['max_moved']
        N = self.cfg.N
        dev = self.device
        cl = self.cell_list

        global_starts = (meta['chain_idx'] * N + meta['bead_start']).int()
        global_ends = (global_starts + meta['n_moved']).int()

        delta_e_out = torch.zeros(B, dtype=torch.float32, device=dev)

        max_nm = int(meta['n_moved'].max().item()) if B > 0 else 0
        if max_nm == 0:
            return delta_e_out, global_starts

        smem_bytes = max_nm * 6 * 4 + 256

        kernel = self.km['delta_e']
        kernel(
            (B,), (256,),
            (self._to_cupy(self.pos_f32.view(-1)),
             self._to_cupy(old_pos),
             self._to_cupy(new_pos),
             self._to_cupy(meta['n_moved']),
             self._to_cupy(global_starts),
             self._to_cupy(global_ends),
             self._to_cupy(delta_e_out),
             self._to_cupy(cl.sorted_order),
             self._to_cupy(cl.cell_starts),
             self._to_cupy(cl.cell_counts),
             self._to_cupy(cl.neighbor_offsets),
             self._to_cupy(self._visited_bitmaps[:B]),
             self._to_cupy(self._visited_lists[:B]),
             self._to_cupy(self._n_visited_out[:B]),
             np.float32(self.box), np.float32(self.inv_box),
             np.float32(self.half_box),
             np.float32(cl._inv_cs), np.int32(cl._nc), np.int32(cl._nc3),
             np.float32(self.r_rep_sq), np.float32(self.rep_e),
             np.int32(mm), np.int32(B), np.int32(self._max_visited)),
            shared_mem=smem_bytes
        )

        return delta_e_out, global_starts

    def _launch_emm(self, old_pos, new_pos, meta):
        B = meta['B']
        mm = meta['max_moved']
        dev = self.device
        n_moved_np = meta['n_moved_np']

        correction = torch.zeros(B, B, dtype=torch.float32, device=dev)

        valid = [i for i in range(B) if n_moved_np[i] > 0]
        if len(valid) < 2:
            return correction

        box = self.box
        inv_box = self.inv_box
        r_rep = math.sqrt(self.r_rep_sq)

        centroids = np.zeros((B, 3), dtype=np.float32)
        radii = np.zeros(B, dtype=np.float32)

        old_cpu = old_pos.cpu().numpy()
        new_cpu = new_pos.cpu().numpy()

        for i in valid:
            nm = int(n_moved_np[i])
            all_pos = np.concatenate([old_cpu[i, :nm], new_cpu[i, :nm]], axis=0)
            centroid = all_pos.mean(axis=0)
            centroids[i] = centroid
            d = all_pos - centroid
            d = d - box * np.round(d * inv_box)
            radii[i] = np.sqrt((d * d).sum(axis=1).max())

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
            (self._to_cupy(old_pos),
             self._to_cupy(new_pos),
             self._to_cupy(meta['n_moved']),
             self._to_cupy(correction),
             self._to_cupy(pair_i_gpu),
             self._to_cupy(pair_j_gpu),
             np.float32(self.box), np.float32(self.inv_box),
             np.float32(self.r_rep_sq), np.float32(self.rep_e),
             np.int32(mm), np.int32(B), np.int32(n_pairs))
        )

        return correction

    def _apply_accepted_moves(self, positions: torch.Tensor, new_pos: torch.Tensor,
                              accepted_indices: list, global_starts: torch.Tensor,
                              meta: dict) -> None:
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
            (self._to_cupy(pos_flat),
             self._to_cupy(new_pos),
             self._to_cupy(accepted_gpu),
             self._to_cupy(global_starts),
             self._to_cupy(meta['n_moved']),
             np.int32(mm), np.int32(n_accepted))
        )
        if not positions.view(-1, 3).is_contiguous():
            positions.view(-1, 3).copy_(pos_flat)

        for idx in accepted_indices:
            ci = meta['chain_idx_np'][idx]
            bs = meta['bead_start_np'][idx]
            nm = meta['n_moved_np'][idx]
            gs = ci * self.cfg.N + bs
            self.pos_f32[gs:gs+nm] = positions.view(-1, 3)[gs:gs+nm].float()

    def perform_sweep(self, positions: torch.Tensor, seg_info: SegmentInfo,
                      stats, rng, skip_pivot: bool = False) -> None:
        cfg = self.cfg
        N = cfg.N
        n_chains = cfg.n_chains
        total_segs = seg_info.total_segments
        kBT = cfg.kBT
        dev = self.device

        self._sync_f32(positions)

        batch_size = GPUFastSweep.MOVE_SIZE
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

        for b_idx, (b_start, b_end) in enumerate(batch_ranges):

            if b_idx % GPUFastSweep.REBUILD_INTERVAL == 0:
                GPUFastSweep._nvtx_push("rebuild_cell_list")
                self._rebuild_cell_list()
                GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("prepare_meta")
            seg_list = []
            for idx in range(b_start, b_end):
                gs = perm[idx]
                seg_list.append((seg_info.seg_chain[gs], seg_info.seg_local[gs]))

            meta = self._prepare_segment_metadata(seg_list, seg_info, N)
            B = meta['B']
            GPUFastSweep._nvtx_pop()

            if B == 0:
                continue

            GPUFastSweep._nvtx_push("segment_proposals")
            old_pos, new_pos = self._launch_segment_proposals(meta, rng)
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("delta_e")
            delta_e_gpu, global_starts = self._launch_delta_e(old_pos, new_pos, meta)
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("emm")
            correction_gpu = self._launch_emm(old_pos, new_pos, meta)
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("accept_d2h_cpu")
            delta_e = delta_e_gpu.cpu().tolist()
            corr = correction_gpu.cpu().tolist()
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("accept_loop_py")
            n_moved_np = meta['n_moved_np']
            move_types = meta['move_types']
            accepted_indices = []

            for i in range(B):
                if n_moved_np[i] == 0:
                    continue

                accepted = GPUFastSweep.metropolis_accept(delta_e[i], kBT, rng)
                stats.record(move_types[i], accepted)

                if not accepted:
                    continue

                accepted_indices.append(i)

                for j in range(i + 1, B):
                    if n_moved_np[j] == 0:
                        continue
                    c = corr[i][j]
                    if c != 0.0:
                        delta_e[j] += c
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("apply_moves")
            self._apply_accepted_moves(positions, new_pos, accepted_indices,
                                       global_starts, meta)
            GPUFastSweep._nvtx_pop()

        if skip_pivot:
            return

        self._sync_f32(positions)
        self._rebuild_cell_list()

        chain_perm = rng.permutation(n_chains).tolist()

        pivot_batch_size = GPUFastSweep.MOVE_SIZE
        for pb_start in range(0, n_chains, pivot_batch_size):
            pb_end = min(pb_start + pivot_batch_size, n_chains)
            batch_chains = chain_perm[pb_start:pb_end]
            Bp = len(batch_chains)

            max_pivot_moved = N - 1 if N > 2 else N

            rand_buf = torch.rand(Bp, 6, dtype=torch.float32, device=dev)

            old_pos_pivot = torch.zeros(Bp, max_pivot_moved, 3,
                                        dtype=torch.float32, device=dev)
            new_pos_pivot = torch.zeros(Bp, max_pivot_moved, 3,
                                        dtype=torch.float32, device=dev)
            rot_start_out = torch.zeros(Bp, dtype=torch.int32, device=dev)
            n_moved_out = torch.zeros(Bp, dtype=torch.int32, device=dev)
            chain_perm_gpu = torch.tensor(batch_chains, dtype=torch.int32, device=dev)

            GPUFastSweep._nvtx_push("pivot_proposal")
            kernel = self.km['pivot_proposal']
            max_disp = self.cfg.l0 * (1.0 + 0.05)
            kernel(
                (Bp,), (1,),
                (self._to_cupy(self.pos_f32),
                 self._to_cupy(old_pos_pivot),
                 self._to_cupy(new_pos_pivot),
                 self._to_cupy(rot_start_out),
                 self._to_cupy(n_moved_out),
                 self._to_cupy(rand_buf),
                 self._to_cupy(chain_perm_gpu),
                 np.float32(self.box), np.float32(self.inv_box),
                 np.float32(self.half_box),
                 np.float32(max_disp),
                 np.int32(N), np.int32(n_chains), np.int32(max_pivot_moved))
            )
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("pivot_meta_d2h")
            rot_starts = rot_start_out.cpu().numpy()
            n_moveds = n_moved_out.cpu().numpy()

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
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("pivot_delta_e")
            delta_e_pivot_gpu, global_starts_pivot = self._launch_delta_e(
                old_pos_pivot, new_pos_pivot, pivot_meta)
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("pivot_emm")
            correction_gpu = self._launch_emm(
                old_pos_pivot, new_pos_pivot, pivot_meta)
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("pivot_accept_d2h_cpu")
            delta_e = delta_e_pivot_gpu.cpu().tolist()
            corr = correction_gpu.cpu().tolist()
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("pivot_accept_loop_py")
            accepted_indices = []
            for i in range(Bp):
                if n_moveds[i] == 0:
                    continue

                accepted = GPUFastSweep.metropolis_accept(delta_e[i], kBT, rng)
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
            GPUFastSweep._nvtx_pop()

            if accepted_indices:
                GPUFastSweep._nvtx_push("pivot_apply_moves")
                n_acc = len(accepted_indices)
                acc_gpu = torch.tensor(accepted_indices, dtype=torch.int32, device=dev)

                apply_kernel = self.km['apply_moves']
                apply_kernel(
                    (n_acc,), (256,),
                    (self._to_cupy(positions.view(-1, 3)),
                     self._to_cupy(new_pos_pivot),
                     self._to_cupy(acc_gpu),
                     self._to_cupy(global_starts_pivot),
                     self._to_cupy(n_moved_out),
                     np.int32(max_pivot_moved), np.int32(n_acc))
                )

                for idx in accepted_indices:
                    ci = batch_chains[idx]
                    bs = int(rot_starts[idx])
                    nm = int(n_moveds[idx])
                    gs = ci * N + bs
                    self.pos_f32[gs:gs+nm] = positions.view(-1, 3)[gs:gs+nm].float()
                GPUFastSweep._nvtx_pop()

    def perform_pivot_phase(self, positions: torch.Tensor, seg_info: SegmentInfo,
                            stats, rng) -> None:
        cfg = self.cfg
        N = cfg.N
        n_chains = cfg.n_chains
        dev = self.device

        self._sync_f32(positions)
        self._rebuild_cell_list()

        chain_perm = rng.permutation(n_chains).tolist()

        pivot_batch_size = GPUFastSweep.MOVE_SIZE
        for pb_start in range(0, n_chains, pivot_batch_size):
            pb_end = min(pb_start + pivot_batch_size, n_chains)
            batch_chains = chain_perm[pb_start:pb_end]
            Bp = len(batch_chains)

            max_pivot_moved = N - 1 if N > 2 else N

            rand_buf = torch.rand(Bp, 6, dtype=torch.float32, device=dev)

            old_pos_pivot = torch.zeros(Bp, max_pivot_moved, 3,
                                        dtype=torch.float32, device=dev)
            new_pos_pivot = torch.zeros(Bp, max_pivot_moved, 3,
                                        dtype=torch.float32, device=dev)
            rot_start_out = torch.zeros(Bp, dtype=torch.int32, device=dev)
            n_moved_out = torch.zeros(Bp, dtype=torch.int32, device=dev)
            chain_perm_gpu = torch.tensor(batch_chains, dtype=torch.int32, device=dev)

            GPUFastSweep._nvtx_push("pivot_proposal")
            kernel = self.km['pivot_proposal']
            max_disp = self.cfg.l0 * (1.0 + 0.05)
            kernel(
                (Bp,), (1,),
                (self._to_cupy(self.pos_f32),
                 self._to_cupy(old_pos_pivot),
                 self._to_cupy(new_pos_pivot),
                 self._to_cupy(rot_start_out),
                 self._to_cupy(n_moved_out),
                 self._to_cupy(rand_buf),
                 self._to_cupy(chain_perm_gpu),
                 np.float32(self.box), np.float32(self.inv_box),
                 np.float32(self.half_box),
                 np.float32(max_disp),
                 np.int32(N), np.int32(n_chains), np.int32(max_pivot_moved))
            )
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("pivot_meta_d2h")
            rot_starts = rot_start_out.cpu().numpy()
            n_moveds = n_moved_out.cpu().numpy()

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
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("pivot_delta_e")
            delta_e_pivot_gpu, global_starts_pivot = self._launch_delta_e(
                old_pos_pivot, new_pos_pivot, pivot_meta)
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("pivot_emm")
            correction_gpu = self._launch_emm(
                old_pos_pivot, new_pos_pivot, pivot_meta)
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("pivot_accept_d2h_cpu")
            delta_e = delta_e_pivot_gpu.cpu().tolist()
            corr = correction_gpu.cpu().tolist()
            GPUFastSweep._nvtx_pop()

            GPUFastSweep._nvtx_push("pivot_accept_loop_py")
            accepted_indices = []
            for i in range(Bp):
                if n_moveds[i] == 0:
                    continue

                accepted = GPUFastSweep.metropolis_accept(delta_e[i], cfg.kBT, rng)
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
            GPUFastSweep._nvtx_pop()

            if accepted_indices:
                GPUFastSweep._nvtx_push("pivot_apply_moves")
                n_acc = len(accepted_indices)
                acc_gpu = torch.tensor(accepted_indices, dtype=torch.int32, device=dev)

                apply_kernel = self.km['apply_moves']
                apply_kernel(
                    (n_acc,), (256,),
                    (self._to_cupy(positions.view(-1, 3)),
                     self._to_cupy(new_pos_pivot),
                     self._to_cupy(acc_gpu),
                     self._to_cupy(global_starts_pivot),
                     self._to_cupy(n_moved_out),
                     np.int32(max_pivot_moved), np.int32(n_acc))
                )

                for idx in accepted_indices:
                    ci = batch_chains[idx]
                    bs = int(rot_starts[idx])
                    nm = int(n_moveds[idx])
                    gs = ci * N + bs
                    self.pos_f32[gs:gs+nm] = positions.view(-1, 3)[gs:gs+nm].float()
                GPUFastSweep._nvtx_pop()
