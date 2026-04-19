"""BatchProposal — batch of B move proposals as contiguous GPU tensors."""

import torch


class BatchProposal:
    __slots__ = ['B', 'old_pos', 'new_pos', 'chain_idx', 'bead_start',
                 'n_moved', 'move_types', 'valid', 'max_moved',
                 '_old_f32', '_new_f32',
                 'chain_idx_t', 'bead_start_t', 'n_moved_t']

    def __init__(self, B: int, max_moved: int, device: torch.device,
                 dtype: torch.dtype = torch.float64):
        self.B = B
        self.max_moved = max_moved
        self.old_pos = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
        self.new_pos = torch.zeros(B, max_moved, 3, dtype=dtype, device=device)
        self.chain_idx = [0] * B
        self.bead_start = [0] * B
        self.n_moved = [0] * B
        self.move_types = [''] * B
        self.valid = [False] * B
        self._old_f32 = None
        self._new_f32 = None
        # Optional GPU-resident mirrors. Populated by GPU-native proposers
        # (Phase 2) so multistep_mc.py can skip the per-batch H2D conversion
        # of the Python lists above. None means "fall back to lists".
        self.chain_idx_t = None
        self.bead_start_t = None
        self.n_moved_t = None

    def get_f32(self):
        if self._old_f32 is None:
            if self.old_pos.dtype == torch.float32:
                self._old_f32 = self.old_pos
                self._new_f32 = self.new_pos
            else:
                self._old_f32 = self.old_pos.float()
                self._new_f32 = self.new_pos.float()
        return self._old_f32, self._new_f32

    def get_gs_nm_tensors(self, N: int, device: torch.device):
        gs = torch.tensor([c * N + b for c, b in
                           zip(self.chain_idx, self.bead_start)],
                          dtype=torch.long, device=device)
        nm = torch.tensor(self.n_moved, dtype=torch.long, device=device)
        return gs, nm
