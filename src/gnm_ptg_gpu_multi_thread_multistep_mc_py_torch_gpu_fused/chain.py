"""ChainState — Calpha + SG positions on a CUDA device (torch tensors).

Initialization is on CPU (numpy SAW grower) for portability, then pushed
to GPU as float32 tensors. Bond rescaling is on CPU (cheap relative to
the MC inner loop).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from .config import SimConfig
from .number_space import NumberSpace


SEG_N_TERMINAL = 0
SEG_C_TERMINAL = 1
SEG_INNER = 2
SEG_BOTH = 3


class SegmentInfo:
    META_CHAIN_IDX = 0
    META_BEAD_START = 1
    META_N_MOVED = 2
    META_SEG_TYPE = 3
    META_ANCHOR_A = 4
    META_ANCHOR_B = 5
    META_NCOLS = 6

    def __init__(self, cfg: SimConfig):
        N = cfg.N
        n_chains = cfg.n_chains
        seg_size = cfg.residues_per_segment
        spc = cfg.n_segments_per_chain
        seg_starts, seg_ends, seg_types = [], [], []
        for s in range(spc):
            ss = s * seg_size
            se = min((s + 1) * seg_size, N)
            seg_starts.append(ss); seg_ends.append(se)
            if spc == 1:
                seg_types.append(SEG_BOTH)
            elif s == 0:
                seg_types.append(SEG_N_TERMINAL)
            elif s == spc - 1:
                seg_types.append(SEG_C_TERMINAL)
            else:
                seg_types.append(SEG_INNER)
        self.seg_starts = seg_starts; self.seg_ends = seg_ends; self.seg_types = seg_types
        self.segs_per_chain = spc
        self.total_segments = n_chains * spc

        rows = np.zeros((self.total_segments, self.META_NCOLS), dtype=np.int64)
        gs = 0
        for c in range(n_chains):
            for s in range(spc):
                stype = seg_types[s]
                ss = seg_starts[s]; se = seg_ends[s]
                if stype == SEG_INNER:
                    a_idx = max(ss - 1, 0); b_idx = min(se, N - 1)
                    ms, me = ss, se
                elif stype in (SEG_N_TERMINAL, SEG_BOTH):
                    a_idx = min(se, N - 1); b_idx = min(se + 1, N - 1)
                    ms, me = 0, se
                else:
                    a_idx = max(ss - 1, 0); b_idx = max(ss - 2, 0)
                    ms, me = ss, N
                rows[gs] = (c, ms, me - ms, stype, a_idx, b_idx)
                gs += 1
        self.table = rows
        self.max_moved_static = max(seg_size, N - seg_size) if spc > 1 else N


class ChainState:

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.gpu_device}")
        self.dtype = torch.float32 if cfg.fp32 else torch.float64
        self.ns = NumberSpace(cfg.box_size, cfg.sigma, self.device, self.dtype)
        self.segments = SegmentInfo(cfg)
        self.ca = torch.zeros(cfg.n_chains, cfg.N, 3, dtype=self.dtype, device=self.device)
        self.sg = torch.zeros(cfg.n_chains, cfg.N, 3, dtype=self.dtype, device=self.device)

    def initialize(self, rng: np.random.Generator) -> None:
        cfg = self.cfg
        ca_np = np.zeros((cfg.n_chains, cfg.N, 3), dtype=np.float64)
        sg_np = np.zeros((cfg.n_chains, cfg.N, 3), dtype=np.float64)
        if cfg.init_method == "random_saw":
            self._init_saw_np(ca_np, rng)
        else:
            self._init_serpentine_np(ca_np)
        self._place_sg_np(ca_np, sg_np, rng)
        self._rescale_np(ca_np, sg_np)
        self.ca.copy_(torch.from_numpy(ca_np.astype(np.float32)).to(self.device))
        self.sg.copy_(torch.from_numpy(sg_np.astype(np.float32)).to(self.device))

    def to_cpu(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.ca.detach().cpu().numpy().astype(np.float64),
                self.sg.detach().cpu().numpy().astype(np.float64))

    def rescale_bonds(self) -> Tuple[float, float]:
        ca_np, sg_np = self.to_cpu()
        cfg = self.cfg
        bond = ca_np[:, 1:, :] - ca_np[:, :-1, :]
        bond -= cfg.box_size * np.round(bond / cfg.box_size)
        L = np.sqrt((bond * bond).sum(axis=-1))
        pre = float(np.abs(L / cfg.l0 - 1.0).max())
        self._rescale_np(ca_np, sg_np)
        self.ca.copy_(torch.from_numpy(ca_np.astype(np.float32)).to(self.device))
        self.sg.copy_(torch.from_numpy(sg_np.astype(np.float32)).to(self.device))
        return 0.0, pre

    # ── numpy helpers ───────────────────────────────────────────────
    def _wrap(self, v):
        ns_box = self.cfg.box_size
        half = ns_box / 2.0
        return ((v + half) % ns_box) - half

    def _init_serpentine_np(self, ca):
        cfg = self.cfg
        ns_box = cfg.box_size
        cpa = math.ceil(cfg.n_chains ** (1.0 / 3.0))
        cell = ns_box / cpa
        sp = cfg.l0
        for c in range(cfg.n_chains):
            gx = c % cpa; gy = (c // cpa) % cpa; gz = (c // (cpa * cpa)) % cpa
            x0 = gx * cell - cfg.half_box + sp
            y0 = gy * cell - cfg.half_box + sp
            z0 = gz * cell - cfg.half_box + sp
            cx, cy, cz = x0, y0, z0
            for i in range(cfg.N):
                ca[c, i] = (self._wrap(cx), self._wrap(cy), self._wrap(cz))
                cz += sp
                if cz > z0 + cell - sp:
                    cz = z0; cx += sp

    def _init_saw_np(self, ca, rng):
        cfg = self.cfg
        ns_box = cfg.box_size
        sp = cfg.l0
        sigma_sq = cfg.sigma ** 2
        for c in range(cfg.n_chains):
            ca[c, 0] = (ns_box * (rng.random() - 0.5),
                        ns_box * (rng.random() - 0.5),
                        ns_box * (rng.random() - 0.5))
            for i in range(1, cfg.N):
                placed = False
                for _ in range(50):
                    u = 2.0 * rng.random() - 1.0
                    v = 2.0 * rng.random() - 1.0
                    s = u * u + v * v
                    if not (1e-10 < s < 1.0): continue
                    f = 2.0 * math.sqrt(1.0 - s)
                    cand = ca[c, i - 1] + np.array([u * f * sp, v * f * sp, (1 - 2 * s) * sp])
                    cand = self._wrap(cand)
                    prev = ca[c, :i]
                    d = cand - prev
                    d -= ns_box * np.round(d / ns_box)
                    if (d * d).sum(axis=-1).min() >= sigma_sq:
                        ca[c, i] = cand
                        placed = True
                        break
                if not placed:
                    ca[c, i] = self._wrap(ca[c, i - 1] + np.array([sp, 0.0, 0.0]))

    def _place_sg_np(self, ca, sg, rng):
        cfg = self.cfg
        ns_box = cfg.box_size
        d = cfg.d_ca_sg
        for c in range(cfg.n_chains):
            for i in range(cfg.N):
                if 0 < i < cfg.N - 1:
                    tang = ca[c, i + 1] - ca[c, i - 1]
                elif i == 0 and cfg.N > 1:
                    tang = ca[c, 1] - ca[c, 0]
                elif i == cfg.N - 1 and cfg.N > 1:
                    tang = ca[c, i] - ca[c, i - 1]
                else:
                    tang = np.array([1.0, 0.0, 0.0])
                tang -= ns_box * np.round(tang / ns_box)
                tn = math.sqrt(float((tang * tang).sum()))
                tang = tang / tn if tn > 1e-12 else np.array([1.0, 0.0, 0.0])
                while True:
                    rv = rng.normal(size=3)
                    proj = float((rv * tang).sum())
                    perp = rv - proj * tang
                    pn = math.sqrt(float((perp * perp).sum()))
                    if pn > 1e-9:
                        perp /= pn; break
                sg[c, i] = self._wrap(ca[c, i] + perp * d)

    def _rescale_np(self, ca, sg):
        cfg = self.cfg
        ns_box = cfg.box_size
        l0 = cfg.l0
        sg_off = sg - ca
        sg_off -= ns_box * np.round(sg_off / ns_box)
        bond = ca[:, 1:, :] - ca[:, :-1, :]
        bond -= ns_box * np.round(bond / ns_box)
        L = np.sqrt((bond * bond).sum(axis=-1, keepdims=True)).clip(min=1e-12)
        dirs = bond / L
        for c in range(cfg.n_chains):
            for i in range(cfg.N - 1):
                ca[c, i + 1] = self._wrap(ca[c, i] + l0 * dirs[c, i])
        sg[...] = self._wrap(ca + sg_off)
