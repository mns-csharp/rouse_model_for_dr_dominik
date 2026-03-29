"""
BatchProposal: keeps proposal data as GPU batch tensors throughout the
propose → energy → accept pipeline.

Eliminates the MoveProposal unpack/repack cycle that causes ~3ms of
Python loop overhead per batch (20 iterations × 150μs per iter).
"""

import torch


class BatchProposal:
    """
    Batch of B move proposals stored as contiguous GPU tensors.

    Replaces list[MoveProposal] to avoid per-proposal Python overhead.
    All tensors stay on GPU until the acceptance loop needs scalar values.
    """
    __slots__ = ['B', 'old_pos', 'new_pos', 'chain_idx', 'bead_start',
                 'n_moved', 'move_types', 'valid', 'max_moved',
                 '_old_f32', '_new_f32']

    def __init__(self, B: int, max_moved: int, device: torch.device,
                 dtype: torch.dtype = torch.float64):
        self.B = B
        self.max_moved = max_moved
        # Native dtype positions (for state updates)
        self.old_pos = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
        self.new_pos = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
        # Metadata (CPU Python lists — small, no overhead)
        self.chain_idx = [0] * B
        self.bead_start = [0] * B
        self.n_moved = [0] * B
        self.move_types = [''] * B
        self.valid = [False] * B
        # Cached FP32 copies for energy computation (lazily created)
        self._old_f32 = None
        self._new_f32 = None

    def get_f32(self):
        """Get FP32 copies of positions for energy computation.
        Cached — only converts once per batch."""
        if self._old_f32 is None:
            if self.old_pos.dtype == torch.float32:
                self._old_f32 = self.old_pos
                self._new_f32 = self.new_pos
            else:
                self._old_f32 = self.old_pos.float()
                self._new_f32 = self.new_pos.float()
        return self._old_f32, self._new_f32

    def get_gs_nm_tensors(self, N: int, device: torch.device):
        """Get global_start and n_moved as GPU tensors for vectorized masking."""
        gs = torch.tensor([c * N + b for c, b in
                           zip(self.chain_idx, self.bead_start)],
                          dtype=torch.long, device=device)
        nm = torch.tensor(self.n_moved, dtype=torch.long, device=device)
        return gs, nm
