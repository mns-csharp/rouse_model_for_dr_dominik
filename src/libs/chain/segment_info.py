"""SegmentInfo — precomputed segment boundaries and types for segmented multi-step MC."""

from rouse_model_python.src.libs.config.simulation_config import SimulationConfig


class SegmentInfo:
    N_TERMINAL = 0
    C_TERMINAL = 1
    INNER = 2
    BOTH = 3  # single-segment chain

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        seg_size = cfg.residues_per_segment
        N = cfg.N
        n_chains = cfg.n_chains

        segs_per_chain = cfg.n_segments_per_chain
        starts, ends, types = [], [], []
        for s in range(segs_per_chain):
            s_start = s * seg_size
            s_end = min((s + 1) * seg_size, N)
            starts.append(s_start)
            ends.append(s_end)
            if segs_per_chain == 1:
                types.append(self.BOTH)
            elif s == 0:
                types.append(self.N_TERMINAL)
            elif s == segs_per_chain - 1:
                types.append(self.C_TERMINAL)
            else:
                types.append(self.INNER)

        self.seg_starts = starts
        self.seg_ends = ends
        self.seg_types = types
        self.segs_per_chain = segs_per_chain
        self.total_segments = n_chains * segs_per_chain

        self.seg_chain = []
        self.seg_local = []
        for c in range(n_chains):
            for s in range(segs_per_chain):
                self.seg_chain.append(c)
                self.seg_local.append(s)

    def get_segment_range(self, chain_idx: int, local_seg: int):
        return self.seg_starts[local_seg], self.seg_ends[local_seg]

    def get_segment_type(self, local_seg: int) -> int:
        return self.seg_types[local_seg]

    def get_hinge_axis_indices(self, local_seg: int, N: int):
        s_start = self.seg_starts[local_seg]
        s_end = self.seg_ends[local_seg]
        axis_start = max(s_start - 1, 0)
        axis_end = min(s_end, N - 1)
        return axis_start, axis_end
