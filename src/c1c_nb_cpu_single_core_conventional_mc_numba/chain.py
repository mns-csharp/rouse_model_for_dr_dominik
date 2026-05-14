"""ChainState + SegmentInfo + initialization + bond rescaler.

SURPASS-alpha layout:
  - ca[n_chains, N, 3]  — Calpha positions (one per residue)
  - sg[n_chains, N, 3]  — side-group positions (one per residue, attached to its Calpha)

The Calpha-Calpha backbone is rigid (length l0). Each SG bead sits at a
fixed offset of length d_ca_sg from its parent Calpha. Both move together
under any MC move (rigid-body rotation of the whole residue group).

Energy interactions are evaluated on a flat 2N-per-chain bead list:
  flat index i = 2 * (chain * N + residue) + (0=Calpha, 1=SG)

Sequence-separation exclusions live in `min_seq_caca`, `min_seq_casg`,
`min_seq_sgsg` on SimConfig; bonded backbone neighbours and intra-residue
Calpha-SG pairs are always skipped.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .config import SimConfig
from .number_space import NumberSpace


# Segment-type enum (matches libs/chain/segment_info.py).
SEG_N_TERMINAL = 0
SEG_C_TERMINAL = 1
SEG_INNER = 2
SEG_BOTH = 3


class SegmentInfo:
    """Precomputed segment table for the residue-level segmentation."""

    META_CHAIN_IDX = 0
    META_BEAD_START = 1
    META_N_MOVED = 2
    META_SEG_TYPE = 3
    META_ANCHOR_A = 4   # local residue index in chain
    META_ANCHOR_B = 5
    META_NCOLS = 6

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        N = cfg.N
        n_chains = cfg.n_chains
        seg_size = cfg.residues_per_segment
        segs_per_chain = cfg.n_segments_per_chain

        seg_starts, seg_ends, seg_types = [], [], []
        for s in range(segs_per_chain):
            s_start = s * seg_size
            s_end = min((s + 1) * seg_size, N)
            seg_starts.append(s_start)
            seg_ends.append(s_end)
            if segs_per_chain == 1:
                seg_types.append(SEG_BOTH)
            elif s == 0:
                seg_types.append(SEG_N_TERMINAL)
            elif s == segs_per_chain - 1:
                seg_types.append(SEG_C_TERMINAL)
            else:
                seg_types.append(SEG_INNER)

        self.seg_starts = seg_starts
        self.seg_ends = seg_ends
        self.seg_types = seg_types
        self.segs_per_chain = segs_per_chain
        self.total_segments = n_chains * segs_per_chain

        # Build table — one row per global segment.
        rows = np.zeros((self.total_segments, self.META_NCOLS), dtype=np.int64)
        gs = 0
        for c in range(n_chains):
            for s in range(segs_per_chain):
                stype = seg_types[s]
                s_start = seg_starts[s]
                s_end = seg_ends[s]
                if stype == SEG_INNER:
                    a_idx = max(s_start - 1, 0)
                    b_idx = min(s_end, N - 1)
                    ms, me = s_start, s_end
                elif stype in (SEG_N_TERMINAL, SEG_BOTH):
                    a_idx = min(s_end, N - 1)
                    b_idx = min(s_end + 1, N - 1)
                    ms, me = 0, s_end
                else:  # SEG_C_TERMINAL
                    a_idx = max(s_start - 1, 0)
                    b_idx = max(s_start - 2, 0)
                    ms, me = s_start, N
                rows[gs, self.META_CHAIN_IDX] = c
                rows[gs, self.META_BEAD_START] = ms
                rows[gs, self.META_N_MOVED] = me - ms
                rows[gs, self.META_SEG_TYPE] = stype
                rows[gs, self.META_ANCHOR_A] = a_idx
                rows[gs, self.META_ANCHOR_B] = b_idx
                gs += 1
        self.table = rows

        if segs_per_chain == 1:
            self.max_moved_static = N
        else:
            self.max_moved_static = max(seg_size, N - seg_size)


class ChainState:
    """Holds Calpha and SG positions plus the box geometry.

    SG offsets are stored relative to the parent Calpha (computed at init);
    after every accepted move the SG bead is translated rigidly with its
    Calpha, so the relative offset is preserved.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.ns = NumberSpace(cfg.box_size, cfg.sigma)
        self.segments = SegmentInfo(cfg)
        # ca[c, i, :], sg[c, i, :]: float64. Stored wrapped into the box.
        self.ca = np.zeros((cfg.n_chains, cfg.N, 3), dtype=np.float64)
        self.sg = np.zeros((cfg.n_chains, cfg.N, 3), dtype=np.float64)

    # ── Initialization ─────────────────────────────────────────────────
    def initialize(self, rng: np.random.Generator):
        method = self.cfg.init_method
        if method == "random_saw":
            self._init_random_saw(rng)
        elif method == "serpentine":
            self._init_serpentine()
        else:
            raise ValueError(f"unknown init_method: {method}")
        self._place_sg_beads(rng)

    def _init_serpentine(self):
        cfg = self.cfg
        ns = self.ns
        n_chains, N = cfg.n_chains, cfg.N
        spacing = cfg.l0
        cpa = math.ceil(n_chains ** (1.0 / 3.0))
        cell_size = cfg.box_size / cpa
        margin = spacing
        for c in range(n_chains):
            gx = c % cpa
            gy = (c // cpa) % cpa
            gz = (c // (cpa * cpa)) % cpa
            x0 = gx * cell_size - ns.half_box + margin
            y0 = gy * cell_size - ns.half_box + margin
            z0 = gz * cell_size - ns.half_box + margin
            x_max = (gx + 1) * cell_size - ns.half_box - margin
            z_max = (gz + 1) * cell_size - ns.half_box - margin
            cx, cy, cz = x0, y0, z0
            z_dir, x_dir = 1, 1
            for i in range(N):
                self.ca[c, i, 0] = ns.wrap_scalar(cx)
                self.ca[c, i, 1] = ns.wrap_scalar(cy)
                self.ca[c, i, 2] = ns.wrap_scalar(cz)
                next_z = cz + z_dir * spacing
                if (z_dir > 0 and next_z <= z_max) or (z_dir < 0 and next_z >= z0):
                    cz = next_z
                else:
                    next_x = cx + x_dir * spacing
                    if (x_dir > 0 and next_x <= x_max) or (x_dir < 0 and next_x >= x0):
                        cx = next_x
                        z_dir = -z_dir
                    else:
                        cy += spacing
                        x_dir = -x_dir
                        z_dir = -z_dir

    def _init_random_saw(self, rng: np.random.Generator):
        cfg = self.cfg
        ns = self.ns
        n_chains, N = cfg.n_chains, cfg.N
        spacing = cfg.l0
        sigma = cfg.sigma
        sigma_sq = sigma * sigma
        max_tries = 200

        cpa = math.ceil(n_chains ** (1.0 / 3.0))
        cell_size = ns.box_size / cpa

        for c in range(n_chains):
            placed = False
            for _ in range(max_tries):
                cx = ns.box_size * (rng.random() - 0.5)
                cy = ns.box_size * (rng.random() - 0.5)
                cz = ns.box_size * (rng.random() - 0.5)
                self._grow_chain_saw(c, cx, cy, cz, spacing, sigma_sq, rng)
                if c == 0 or not self._chain_overlaps_prev(c, sigma_sq):
                    placed = True
                    break
            if not placed:
                gx = c % cpa
                gy = (c // cpa) % cpa
                gz = (c // (cpa * cpa)) % cpa
                cx = (gx + 0.5) * cell_size - ns.half_box
                cy = (gy + 0.5) * cell_size - ns.half_box
                cz = (gz + 0.5) * cell_size - ns.half_box
                self._grow_chain_saw(c, cx, cy, cz, spacing, sigma_sq, rng)

    def _grow_chain_saw(self, c, cx, cy, cz, spacing, sigma_sq, rng):
        N = self.cfg.N
        ns = self.ns
        max_dir = 50
        max_back = max(4 * N, 100)
        self.ca[c, 0, 0] = ns.wrap_scalar(cx)
        self.ca[c, 0, 1] = ns.wrap_scalar(cy)
        self.ca[c, 0, 2] = ns.wrap_scalar(cz)
        if N <= 1:
            return
        i = 1
        backtrack = 0
        while i < N:
            placed = False
            for _ in range(max_dir):
                u = 2.0 * rng.random() - 1.0
                v = 2.0 * rng.random() - 1.0
                s = u * u + v * v
                if not (1e-10 < s < 1.0):
                    continue
                factor = 2.0 * math.sqrt(1.0 - s)
                cand_x = ns.wrap_scalar(self.ca[c, i - 1, 0] + u * factor * spacing)
                cand_y = ns.wrap_scalar(self.ca[c, i - 1, 1] + v * factor * spacing)
                cand_z = ns.wrap_scalar(self.ca[c, i - 1, 2] + (1.0 - 2.0 * s) * spacing)
                cand = np.array([cand_x, cand_y, cand_z], dtype=np.float64)
                if self._intra_chain_min_dist_sq(c, i, cand) >= sigma_sq:
                    self.ca[c, i, 0] = cand_x
                    self.ca[c, i, 1] = cand_y
                    self.ca[c, i, 2] = cand_z
                    i += 1
                    placed = True
                    break
            if placed:
                continue
            if i <= 1:
                # Place arbitrarily; rare, accept the overlap.
                self.ca[c, 1, 0] = ns.wrap_scalar(self.ca[c, 0, 0] + spacing)
                self.ca[c, 1, 1] = self.ca[c, 0, 1]
                self.ca[c, 1, 2] = self.ca[c, 0, 2]
                i = 2
                continue
            i -= 1
            backtrack += 1
            if backtrack > max_back:
                # Bail with current placement; bond rescaler will tidy up.
                while i < N:
                    self.ca[c, i, 0] = ns.wrap_scalar(self.ca[c, i - 1, 0] + spacing)
                    self.ca[c, i, 1] = self.ca[c, i - 1, 1]
                    self.ca[c, i, 2] = self.ca[c, i - 1, 2]
                    i += 1
                return

    def _intra_chain_min_dist_sq(self, c, i, cand):
        if i <= 0:
            return float("inf")
        prev = self.ca[c, :i, :]
        d2 = self.ns.mic_dist_sq(prev, cand)
        return float(d2.min())

    def _chain_overlaps_prev(self, c, sigma_sq):
        if c == 0:
            return False
        chain_c = self.ca[c]                 # [N, 3]
        prev = self.ca[:c].reshape(-1, 3)    # [(c)*N, 3]
        a = chain_c[:, None, :]
        b = prev[None, :, :]
        d2 = self.ns.mic_dist_sq(a, b)
        return bool((d2 < sigma_sq).any())

    def _place_sg_beads(self, rng: np.random.Generator):
        """Place each SG bead at a fixed offset of length d_ca_sg
        from its parent Calpha. Direction is chosen perpendicular to the
        local backbone tangent so SG sits "off to the side" of the chain;
        for terminal residues a random unit vector is used."""
        cfg = self.cfg
        ns = self.ns
        d = cfg.d_ca_sg
        for c in range(cfg.n_chains):
            for i in range(cfg.N):
                # Local tangent
                if 0 < i < cfg.N - 1:
                    tang = ns.mic_delta(self.ca[c, i - 1], self.ca[c, i + 1])
                elif i == 0 and cfg.N > 1:
                    tang = ns.mic_delta(self.ca[c, 0], self.ca[c, 1])
                elif i == cfg.N - 1 and cfg.N > 1:
                    tang = ns.mic_delta(self.ca[c, i - 1], self.ca[c, i])
                else:
                    tang = np.array([1.0, 0.0, 0.0])
                tn = math.sqrt(float((tang * tang).sum()))
                if tn < 1e-12:
                    tang = np.array([1.0, 0.0, 0.0])
                else:
                    tang = tang / tn
                # Pick a random unit vector and project out tangent component.
                while True:
                    rv = rng.normal(size=3)
                    proj = float((rv * tang).sum())
                    perp = rv - proj * tang
                    pn = math.sqrt(float((perp * perp).sum()))
                    if pn > 1e-9:
                        perp /= pn
                        break
                offset = perp * d
                self.sg[c, i] = ns.wrap(self.ca[c, i] + offset)

    # ── Bond integrity ────────────────────────────────────────────────
    def max_bond_stretch(self) -> float:
        ns = self.ns
        bond = ns.mic_delta(self.ca[:, :-1, :], self.ca[:, 1:, :])
        L = np.sqrt((bond * bond).sum(axis=-1))
        return float(np.abs(L / self.cfg.l0 - 1.0).max())

    def rescale_bonds(self) -> Tuple[float, float]:
        """Snap each Calpha-Calpha bond back to l0 (preserving direction).

        Walks each chain from bead 0 and sets bond i+1 to bead i + l0 * d_hat(orig).
        Then re-attaches each SG bead by preserving its (Calpha-relative) offset.
        Returns (max_drift_A, max_stretch_pre)."""
        cfg = self.cfg
        ns = self.ns
        l0 = cfg.l0
        box = cfg.box_size
        inv_box = 1.0 / box
        half = ns.half_box

        # Capture pre-rescale stretch.
        pre = self.max_bond_stretch()

        # Capture all SG offsets relative to their parent Calpha (MIC).
        sg_off = ns.mic_delta(self.ca, self.sg)  # [n_chains, N, 3]

        # Capture original bond directions (vectorized).
        bond = ns.mic_delta(self.ca[:, :-1, :], self.ca[:, 1:, :])
        L = np.sqrt((bond * bond).sum(axis=-1, keepdims=True)).clip(min=1e-12)
        dirs = bond / L

        max_drift = 0.0
        N = cfg.N
        for c in range(cfg.n_chains):
            for i in range(N - 1):
                new_x = self.ca[c, i, 0] + l0 * dirs[c, i, 0]
                new_y = self.ca[c, i, 1] + l0 * dirs[c, i, 1]
                new_z = self.ca[c, i, 2] + l0 * dirs[c, i, 2]
                new_x -= box * math.floor((new_x + half) * inv_box)
                new_y -= box * math.floor((new_y + half) * inv_box)
                new_z -= box * math.floor((new_z + half) * inv_box)
                drx = new_x - self.ca[c, i + 1, 0]
                dry = new_y - self.ca[c, i + 1, 1]
                drz = new_z - self.ca[c, i + 1, 2]
                drx -= box * round(drx * inv_box)
                dry -= box * round(dry * inv_box)
                drz -= box * round(drz * inv_box)
                drift = math.sqrt(drx * drx + dry * dry + drz * drz)
                if drift > max_drift:
                    max_drift = drift
                self.ca[c, i + 1, 0] = new_x
                self.ca[c, i + 1, 1] = new_y
                self.ca[c, i + 1, 2] = new_z
        # Re-attach SG by replaying captured offsets onto wrapped Calpha.
        self.sg = ns.wrap(self.ca + sg_off)
        return max_drift, pre

    # ── Flat-bead view used by the energy module ──────────────────────
    def get_flat_beads(self) -> np.ndarray:
        """Return [n_chains*N*2, 3] — interleaved [Calpha_0, SG_0, Calpha_1, SG_1, ...].

        Bead type for flat index `f` is `f & 1` (0 = Calpha, 1 = SG).
        Residue index for flat index `f` is `f >> 1`.
        """
        cfg = self.cfg
        flat = np.empty((cfg.n_chains * cfg.N * 2, 3), dtype=np.float64)
        flat[0::2] = self.ca.reshape(-1, 3)
        flat[1::2] = self.sg.reshape(-1, 3)
        return flat
