"""TrajectoryWriter — PyMOL-compatible per-sweep trajectory output.

Attached to a RouseSimulation / FastRouseSimulation / GPURouseSimulation via
the `snapshot_collector` constructor hook. Implements the duck-typed collector
API that the simulation loops already call:

    should_capture(sweep_idx, phase) -> bool   (cheap pre-check)
    capture(positions, sweep_idx, phase)       (writes a frame)
    close()                                    (flushes handle)

Supported formats:
    pdb  — multi-MODEL PDB with CRYST1 + per-chain CONECT backbone records.
           PyMOL loads it directly (`load traj.pdb`) and draws each chain as
           a connected polymer; the MODEL slider animates frames.
    xyz  — multi-frame XYZ. Smaller, no bonds (PyMOL infers by distance).
    none — disable trajectory output.

Coordinates are in Ångström (same units as ChainState.positions). When unwrap
is true (default), periodic-boundary wrapping is reversed per chain so
backbone bonds don't span the simulation box in PyMOL.
"""

import os
from typing import Optional

import numpy as np
import torch


_PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MAX_CHAINS_IN_TRAJECTORY = len(_PDB_CHAIN_IDS)


class TrajectoryWriter:
    def __init__(
        self,
        output_path: str,
        fmt: str,
        stride: int,
        phases: set,
        n_chains: int,
        N: int,
        box_size: float,
        unwrap: bool = True,
    ):
        self.output_path = output_path
        self.fmt = fmt
        self.stride = max(1, int(stride))
        self.phases = set(phases)
        self.n_chains = n_chains
        self.n_chains_in_traj = min(n_chains, MAX_CHAINS_IN_TRAJECTORY)
        self.N = N
        self.box_size = float(box_size)
        self.half_box = 0.5 * self.box_size
        self.unwrap = unwrap
        self._frame_idx = 0
        self._fh = None

        total_beads = self.n_chains_in_traj * N
        if fmt == "pdb" and total_beads > 99999:
            raise ValueError(
                f"PDB atom serial is 5 digits (max 99999) but "
                f"n_chains_in_traj*N = {self.n_chains_in_traj}*{N} = {total_beads}. "
                f"Use --traj_format xyz for systems this large."
            )

        if fmt == "pdb":
            self._strategy = _PDBFrameWriter(self.n_chains_in_traj, N, box_size)
        elif fmt == "xyz":
            self._strategy = _XYZFrameWriter(self.n_chains_in_traj, N)
        else:
            raise ValueError(f"Unknown trajectory format: {fmt!r}")

    def should_capture(self, sweep_idx: int, phase: str) -> bool:
        if phase not in self.phases:
            return False
        return (sweep_idx % self.stride) == 0

    def capture(self, positions, sweep_idx: int, phase: str) -> None:
        if not self.should_capture(sweep_idx, phase):
            return

        if isinstance(positions, torch.Tensor):
            arr = positions.detach().cpu().numpy()
        else:
            arr = np.asarray(positions)
        arr = np.ascontiguousarray(arr[:self.n_chains_in_traj], dtype=np.float64).copy()

        if self.unwrap:
            self._unwrap_chains(arr)

        if self._fh is None:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            self._fh = open(self.output_path, "w")
            self._strategy.write_header(self._fh)

        self._strategy.write_frame(self._fh, arr, self._frame_idx, sweep_idx, phase)
        self._frame_idx += 1

    def close(self) -> None:
        if self._fh is not None:
            self._strategy.write_footer(self._fh)
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def _unwrap_chains(self, pos: np.ndarray) -> None:
        box = self.box_size
        half = self.half_box
        for c in range(self.n_chains_in_traj):
            for i in range(1, self.N):
                for ax in range(3):
                    d = pos[c, i, ax] - pos[c, i - 1, ax]
                    if d > half:
                        pos[c, i:, ax] -= box
                    elif d < -half:
                        pos[c, i:, ax] += box

    @classmethod
    def add_cli_args(cls, parser) -> None:
        parser.add_argument(
            "--traj_format",
            type=str,
            default="pdb",
            choices=["pdb", "xyz", "none"],
            help="Trajectory output format. 'none' disables. Default: pdb.",
        )
        parser.add_argument(
            "--traj_stride",
            type=int,
            default=0,
            help="Frames between recorded sweeps. 0 = auto-pick to target "
                 "~100 frames over eq+prod. Default: 0.",
        )
        parser.add_argument(
            "--traj_phases",
            type=str,
            default="eq,prod",
            help="Comma-separated subset of {eq,prod} to record. Default: eq,prod.",
        )

    @classmethod
    def from_args(cls, args, cfg, traj_dir: str, run_label: str) -> Optional["TrajectoryWriter"]:
        fmt = getattr(args, "traj_format", "pdb")
        if fmt == "none":
            return None

        stride = getattr(args, "traj_stride", 0)
        if stride <= 0:
            total_sweeps = int(cfg.eq_sweeps) + int(cfg.prod_sweeps)
            stride = max(1, total_sweeps // 100)

        phases_raw = getattr(args, "traj_phases", "eq,prod")
        phases = {p.strip() for p in phases_raw.split(",") if p.strip()}

        ext = ".pdb" if fmt == "pdb" else ".xyz"
        output_path = os.path.join(traj_dir, f"{run_label}{ext}")

        return cls(
            output_path=output_path,
            fmt=fmt,
            stride=stride,
            phases=phases,
            n_chains=cfg.n_chains,
            N=cfg.N,
            box_size=cfg.box_size,
        )


class _PDBFrameWriter:
    def __init__(self, n_chains: int, N: int, box_size: float):
        self.n_chains = n_chains
        self.N = N
        self.box_size = float(box_size)

    def write_header(self, fh) -> None:
        b = self.box_size
        fh.write(
            f"CRYST1{b:9.3f}{b:9.3f}{b:9.3f}  90.00  90.00  90.00 P 1           1\n"
        )
        for c in range(self.n_chains):
            base = c * self.N + 1
            for i in range(self.N - 1):
                a = base + i
                b2 = base + i + 1
                fh.write(f"CONECT{a:5d}{b2:5d}\n")
                fh.write(f"CONECT{b2:5d}{a:5d}\n")

    def write_frame(self, fh, pos: np.ndarray, frame_idx: int, sweep_idx: int, phase: str) -> None:
        fh.write(f"MODEL     {frame_idx + 1:4d}\n")
        fh.write(f"REMARK   1 sweep={sweep_idx} phase={phase}\n")
        for c in range(self.n_chains):
            chain_id = _PDB_CHAIN_IDS[c]
            base = c * self.N + 1
            for i in range(self.N):
                serial = base + i
                resseq = i + 1
                x, y, z = pos[c, i, 0], pos[c, i, 1], pos[c, i, 2]
                fh.write(
                    f"ATOM  {serial:5d}  CA  BEA {chain_id}{resseq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            last_serial = base + self.N - 1
            last_resseq = self.N
            fh.write(
                f"TER   {last_serial + 1:5d}      BEA {chain_id}{last_resseq:4d}\n"
            )
        fh.write("ENDMDL\n")

    def write_footer(self, fh) -> None:
        fh.write("END\n")


class _XYZFrameWriter:
    def __init__(self, n_chains: int, N: int):
        self.n_chains = n_chains
        self.N = N
        self.total_beads = n_chains * N

    def write_header(self, fh) -> None:
        pass

    def write_frame(self, fh, pos: np.ndarray, frame_idx: int, sweep_idx: int, phase: str) -> None:
        fh.write(f"{self.total_beads}\n")
        fh.write(f"frame={frame_idx} sweep={sweep_idx} phase={phase}\n")
        for c in range(self.n_chains):
            for i in range(self.N):
                x, y, z = pos[c, i, 0], pos[c, i, 1], pos[c, i, 2]
                fh.write(f"C {x:.3f} {y:.3f} {z:.3f}\n")

    def write_footer(self, fh) -> None:
        pass
