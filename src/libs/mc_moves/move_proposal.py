"""MoveProposal — result of a proposed MC move."""

import torch


class MoveProposal:
    __slots__ = ['chain_idx', 'bead_start', 'bead_end',
                 'old_positions', 'new_positions', 'move_type']

    def __init__(self, chain_idx: int, bead_start: int, bead_end: int,
                 old_positions: torch.Tensor, new_positions: torch.Tensor,
                 move_type: str):
        self.chain_idx = chain_idx
        self.bead_start = bead_start
        self.bead_end = bead_end
        self.old_positions = old_positions
        self.new_positions = new_positions
        self.move_type = move_type

    @property
    def n_moved(self) -> int:
        return self.bead_end - self.bead_start
