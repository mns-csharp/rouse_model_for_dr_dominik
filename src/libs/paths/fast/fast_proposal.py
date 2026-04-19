"""FastProposal — lightweight move proposal container using numpy arrays."""


class FastProposal:
    __slots__ = ['chain_idx', 'bead_start', 'n_moved',
                 'old_pos', 'new_pos', 'move_type']

    def __init__(self, chain_idx: int, bead_start: int, n_moved: int,
                 old_pos, new_pos, move_type: str):
        self.chain_idx = chain_idx
        self.bead_start = bead_start
        self.n_moved = n_moved
        self.old_pos = old_pos
        self.new_pos = new_pos
        self.move_type = move_type
