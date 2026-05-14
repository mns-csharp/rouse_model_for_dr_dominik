"""WorkerPool — long-running torch.multiprocessing workers for delta-E.

Lifecycle is owned by Simulation:
  - Pool is built once at run start (after state moves into SharedMemory).
  - Each batch dispatches W tasks to the input queue and gathers W results
    from the output queue. Workers stay alive between batches.
  - Pool.close() is called from Simulation.run_production / __del__ to
    signal the workers to exit and to release SharedMemory blocks.

Per-batch IPC: only the proposal arrays for each worker's slice are sent
(numpy arrays through the queue, pickled by mp). State is shared memory
and is not sent per-batch.
"""

from __future__ import annotations

import logging
import os
import sys
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _shm_unlink_safe(name: str) -> None:
    try:
        SharedMemory(name=name).unlink()
    except (FileNotFoundError, Exception):
        pass


class WorkerPool:
    """Manages W worker processes that compute delta-E slices in parallel.

    State sharing: at construction, callers must already have moved
    `state.ca` and `state.sg` into SharedMemory blocks (see
    `share_state_inplace`). The pool just records the SharedMemory names
    and passes them to workers at spawn time.
    """

    def __init__(self, n_workers: int,
                 ca_shm_name: str, sg_shm_name: str,
                 state_shape, state_dtype_str: str,
                 cfg_payload: dict):
        # Use torch.multiprocessing so PyTorch's CPU allocator plays nicely.
        # On Windows this enforces 'spawn'. Module-level worker_loop is
        # picklable (defined in worker.py).
        import torch.multiprocessing as tmp

        self.n_workers = int(n_workers)
        if self.n_workers < 1:
            raise ValueError("WorkerPool requires n_workers >= 1")

        ctx = tmp.get_context("spawn")
        self._ctx = ctx
        self._in_qs = [ctx.Queue() for _ in range(self.n_workers)]
        self._out_q = ctx.Queue()
        self._procs = []

        from .worker import worker_loop
        for rank in range(self.n_workers):
            p = ctx.Process(
                target=worker_loop,
                args=(rank, self._in_qs[rank], self._out_q,
                      ca_shm_name, sg_shm_name,
                      tuple(state_shape), state_dtype_str, dict(cfg_payload)),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
        self._closed = False
        logger.info("WorkerPool started with %d workers", self.n_workers)

    def compute_delta_e(self, batch_id: int,
                        old_ca: torch.Tensor, new_ca: torch.Tensor,
                        old_sg: torch.Tensor, new_sg: torch.Tensor,
                        n_moved: torch.Tensor,
                        chain_idx_long: torch.Tensor,
                        bead_start_long: torch.Tensor) -> torch.Tensor:
        """Scatter B proposals across workers; return concatenated delta-E."""
        from .mc import _worker_slice_indices

        B = int(old_ca.shape[0])
        slices = _worker_slice_indices(B, self.n_workers)

        # Numpy views (no copy) of the per-batch proposal tensors.
        old_ca_np = old_ca.numpy()
        new_ca_np = new_ca.numpy()
        old_sg_np = old_sg.numpy()
        new_sg_np = new_sg.numpy()
        n_moved_np = n_moved.numpy().astype(np.int64)
        chain_idx_np = chain_idx_long.numpy().astype(np.int64)
        bead_start_np = bead_start_long.numpy().astype(np.int64)

        # Dispatch: one task per non-empty slice.
        for w, (bs, be) in enumerate(slices):
            self._in_qs[w].put({
                'cmd': 'compute',
                'batch_id': batch_id,
                'slice': (bs, be),
                'old_ca': old_ca_np[bs:be].copy(),
                'new_ca': new_ca_np[bs:be].copy(),
                'old_sg': old_sg_np[bs:be].copy(),
                'new_sg': new_sg_np[bs:be].copy(),
                'n_moved': n_moved_np[bs:be].copy(),
                'chain_idx': chain_idx_np[bs:be].copy(),
                'bead_start': bead_start_np[bs:be].copy(),
            })

        # Gather: read W results from out_q, place each into its slice.
        delta_e = np.zeros(B, dtype=np.float64)
        n_remaining = len(slices)
        while n_remaining > 0:
            res = self._out_q.get()
            if int(res['batch_id']) != batch_id:
                # Late result from a prior batch — should not happen if we
                # gather all W per batch, but be defensive.
                continue
            bs, be = res['slice']
            de_slice = res['delta_e']
            # delta_e tensor in worker is created from energy.batch_delta_e_torch
            # which preserves dtype of old_ca. The proposal tensors are float64,
            # so de_slice is float64.
            delta_e[bs:be] = de_slice
            n_remaining -= 1

        return torch.from_numpy(delta_e)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for q in self._in_qs:
            try:
                q.put({'cmd': 'stop'})
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        logger.info("WorkerPool stopped")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def share_state_inplace(state, sim_id: str):
    """Move state.ca / state.sg into named SharedMemory blocks.

    Returns (ca_shm, sg_shm, ca_name, sg_name, shape, dtype_str). The
    caller (Simulation) is responsible for calling .close()/.unlink() on
    the SharedMemory objects when the simulation ends.
    """
    ca_arr = np.ascontiguousarray(state.ca)
    sg_arr = np.ascontiguousarray(state.sg)
    if ca_arr.dtype != sg_arr.dtype or ca_arr.shape != sg_arr.shape:
        raise ValueError("state.ca and state.sg must share dtype + shape")

    pid = os.getpid()
    ca_name = f"indep_mc_{sim_id}_{pid}_ca"
    sg_name = f"indep_mc_{sim_id}_{pid}_sg"
    _shm_unlink_safe(ca_name)
    _shm_unlink_safe(sg_name)

    ca_shm = SharedMemory(create=True, size=int(ca_arr.nbytes), name=ca_name)
    sg_shm = SharedMemory(create=True, size=int(sg_arr.nbytes), name=sg_name)
    ca_view = np.ndarray(ca_arr.shape, dtype=ca_arr.dtype, buffer=ca_shm.buf)
    sg_view = np.ndarray(sg_arr.shape, dtype=sg_arr.dtype, buffer=sg_shm.buf)
    ca_view[:] = ca_arr
    sg_view[:] = sg_arr

    # Replace the state arrays with the shared-memory-backed views. The
    # rest of the codebase mutates state.ca / state.sg in place, and these
    # views write through to the same physical pages every worker sees.
    state.ca = ca_view
    state.sg = sg_view

    return ca_shm, sg_shm, ca_name, sg_name, tuple(ca_arr.shape), str(ca_arr.dtype)


def cfg_payload_for_worker(cfg) -> dict:
    """Pull just the constants the worker's energy call needs."""
    return {
        'N': int(cfg.N),
        'box': float(cfg.box_size),
        'r_rep_sq': float(cfg.r_rep_sq),
        'r_max_sq': float(cfg.r_max_sq),
        'rep_e': float(cfg.repulsive_energy),
        'contact_e': float(cfg.contact_energy),
        'min_caca': int(cfg.min_seq_caca),
        'min_casg': int(cfg.min_seq_casg),
        'min_sgsg': int(cfg.min_seq_sgsg),
    }
