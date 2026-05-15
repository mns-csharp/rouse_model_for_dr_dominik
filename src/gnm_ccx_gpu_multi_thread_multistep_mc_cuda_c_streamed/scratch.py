"""GPU scratch buffers + cached cupy views.

Per-sweep persistent torch tensors plus their cached `cupy` dlpack views.
Per-call dlpack interop is the second-biggest cuda_c overhead (15 + 10 =
~25 from_dlpack calls per Metropolis step at ~115 us each); creating the
views once at sim init eliminates this entirely.

Buffers cover:
  - propose / delta_e per-call output buffers (B=1 sized)
  - the full chain state (ca, sg) and per-sweep meta-table columns
  - one float32 random per segment, drawn host-side once per sweep
"""

from __future__ import annotations

import numpy as np
import torch


class GpuScratch:

    def __init__(self, state, total_segments: int, M: int, B: int = 1):
        device = state.device
        dtype = state.dtype
        self.M = int(M)
        self.B = int(B)
        self.total_segments = int(total_segments)

        self.old_ca = torch.zeros(B, M, 3, dtype=dtype, device=device)
        self.new_ca = torch.zeros(B, M, 3, dtype=dtype, device=device)
        self.old_sg = torch.zeros(B, M, 3, dtype=dtype, device=device)
        self.new_sg = torch.zeros(B, M, 3, dtype=dtype, device=device)
        self.n_moved_out = torch.zeros(B, dtype=torch.int64, device=device)
        self.move_type = torch.zeros(B, dtype=torch.int64, device=device)
        self.delta_e = torch.zeros(B, dtype=dtype, device=device)

        self.rand_buf = torch.zeros(total_segments, dtype=torch.float32, device=device)
        self.rand_accept = torch.zeros(total_segments, dtype=torch.float32, device=device)

        self.attempted_counts = torch.zeros(3, dtype=torch.int64, device=device)
        self.accepted_counts = torch.zeros(3, dtype=torch.int64, device=device)

        self.perm_buf = torch.zeros(total_segments, dtype=torch.int64, device=device)

        # full meta table on device, transposed so each column is contiguous.
        # cols: (chain_idx, bead_start, n_moved, seg_type, anchor_a, anchor_b)
        table_np = state.segments.table  # int64 [total_segs, 6]
        self.table_full = torch.from_numpy(table_np).to(device).contiguous()  # [total_segs, 6]
        self.col_chain = self.table_full[:, 0].contiguous()
        self.col_bead = self.table_full[:, 1].contiguous()
        self.col_nmov = self.table_full[:, 2].contiguous()
        self.col_seg = self.table_full[:, 3].contiguous()
        self.col_aa = self.table_full[:, 4].contiguous()
        self.col_ab = self.table_full[:, 5].contiguous()

        self._cp_views: dict = {}

    def cp(self, name: str):
        cv = self._cp_views.get(name)
        if cv is None:
            import cupy as cp
            t = getattr(self, name)
            cv = cp.from_dlpack(t)
            self._cp_views[name] = cv
        return cv

    def cp_state_ca(self, ca_t: torch.Tensor):
        cv = self._cp_views.get("__state_ca__")
        if cv is None:
            import cupy as cp
            cv = cp.from_dlpack(ca_t.contiguous().view(-1, 3))
            self._cp_views["__state_ca__"] = cv
        return cv

    def cp_state_sg(self, sg_t: torch.Tensor):
        cv = self._cp_views.get("__state_sg__")
        if cv is None:
            import cupy as cp
            cv = cp.from_dlpack(sg_t.contiguous().view(-1, 3))
            self._cp_views["__state_sg__"] = cv
        return cv

    def fill_rand(self, rng: np.random.Generator) -> None:
        n = self.total_segments
        host = rng.random(2 * n).astype(np.float32, copy=False)
        host_angle = host[:n]
        host_accept = host[n:]
        self.rand_buf.copy_(torch.from_numpy(host_angle), non_blocking=True)
        self.rand_accept.copy_(torch.from_numpy(host_accept), non_blocking=True)

    def reset_counters(self) -> None:
        self.attempted_counts.zero_()
        self.accepted_counts.zero_()
