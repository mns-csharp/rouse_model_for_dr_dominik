"""Long-running worker process — computes delta-E slices in parallel.

Architecture (see mc.py / simulation.py for the call site):

  Main process:
    - Owns ChainState. After initialize() the chain state is moved into
      shared memory so workers can read it without per-batch IPC.
    - Generates B proposals per batch via propose_batch_torch (still serial:
      proposal generation is cheap and avoids extra IPC).
    - Sends each worker its slice of the batch via in_q.
    - Sequentially computes the full BxB correction matrix on the main
      process (correction is BxB pair-wise; parallelizing it across
      processes would require moving slices of the proposal tensors and is
      not worth the IPC overhead at typical B).
    - Gathers per-slice delta_e from out_q and runs the rank-1 fused accept.

  Worker process:
    - Attaches to the SharedMemory blocks for state.ca / state.sg once at
      startup (no per-batch attach cost).
    - Loops on in_q tasks. Each task carries the proposal arrays for the
      worker's slice (sliced on the main side to keep IPC small). Returns
      delta_e for the slice via out_q.

This is the "torch.multiprocessing for multistep" path the user picked
(asymmetric to conventional, which uses intra-op threads only).
"""

from __future__ import annotations

import logging
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Tuple

import numpy as np


def worker_loop(rank: int,
                in_q,
                out_q,
                ca_shm_name: str,
                sg_shm_name: str,
                state_shape: Tuple[int, int, int],
                state_dtype_str: str,
                cfg_payload: Dict[str, Any]) -> None:
    """Long-running worker loop. Imports torch lazily so spawn picks up clean state."""
    import torch
    from .energy import batch_delta_e_torch

    # Workers run single-threaded — the pool itself supplies the parallelism.
    # Without this, W workers each spawning torch.get_num_threads() OMP threads
    # oversubscribes the CPU and tanks throughput.
    torch.set_num_threads(1)

    np_dtype = np.dtype(state_dtype_str)
    ca_shm = SharedMemory(name=ca_shm_name)
    sg_shm = SharedMemory(name=sg_shm_name)
    ca_view = np.ndarray(state_shape, dtype=np_dtype, buffer=ca_shm.buf)
    sg_view = np.ndarray(state_shape, dtype=np_dtype, buffer=sg_shm.buf)

    N = int(cfg_payload['N'])
    box = float(cfg_payload['box'])
    r_rep_sq = float(cfg_payload['r_rep_sq'])
    r_max_sq = float(cfg_payload['r_max_sq'])
    rep_e = float(cfg_payload['rep_e'])
    contact_e = float(cfg_payload['contact_e'])
    min_caca = int(cfg_payload['min_caca'])
    min_casg = int(cfg_payload['min_casg'])
    min_sgsg = int(cfg_payload['min_sgsg'])

    try:
        while True:
            task = in_q.get()
            if task is None:
                break
            cmd = task.get('cmd')
            if cmd == 'stop':
                break
            if cmd != 'compute':
                continue

            ca_t = torch.from_numpy(ca_view)
            sg_t = torch.from_numpy(sg_view)

            old_ca = torch.from_numpy(task['old_ca'])
            new_ca = torch.from_numpy(task['new_ca'])
            old_sg = torch.from_numpy(task['old_sg'])
            new_sg = torch.from_numpy(task['new_sg'])
            n_moved = torch.from_numpy(task['n_moved'])
            chain_idx = torch.from_numpy(task['chain_idx'])
            bead_start = torch.from_numpy(task['bead_start'])

            de = batch_delta_e_torch(
                old_ca, new_ca, old_sg, new_sg, n_moved,
                chain_idx, bead_start,
                ca_t, sg_t,
                N, box,
                r_rep_sq, r_max_sq, rep_e, contact_e,
                min_caca, min_casg, min_sgsg,
            )

            out_q.put({
                'rank': rank,
                'batch_id': int(task['batch_id']),
                'slice': tuple(task['slice']),
                'delta_e': de.numpy(),
            })
    except KeyboardInterrupt:
        pass
    finally:
        ca_shm.close()
        sg_shm.close()
