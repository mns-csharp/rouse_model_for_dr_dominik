"""I/O helpers -- TSV writer, PDB trajectory writer, JSON summary."""

from __future__ import annotations

import csv
import json
import os
from typing import Iterable, Sequence


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_tsv(path: str, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def write_sweep_walls_tsv(path: str, sweep_walls_ms, kernel_walls_ms) -> None:
    """Per-sweep wall-time series. Columns: sweep, wall_ms, kernel_ms.

    `kernel_walls_ms` is zero for backends without per-kernel timing.
    """
    n = len(sweep_walls_ms)
    rows = [[i + 1, float(sweep_walls_ms[i]), float(kernel_walls_ms[i])] for i in range(n)]
    write_tsv(path, ["sweep", "wall_ms", "kernel_ms"], rows)


def write_json(path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def pdb_open_trajectory(path: str, remark: str = ""):
    """Open a multi-MODEL PDB trajectory file for streaming MODEL appends.

    The file handle returned is owned by the caller; close with
    `pdb_close_trajectory`. A REMARK header line is written if provided.
    """
    ensure_dir(os.path.dirname(path) or ".")
    fh = open(path, "w", encoding="utf-8")
    if remark:
        fh.write(f"REMARK   1 {remark}\n")
    return fh


def pdb_write_model(fh, ca, sg, model_idx: int) -> None:
    """Append one MODEL block (CA + SG records, TER per chain, ENDMDL) to `fh`."""
    n_chains, N, _ = ca.shape
    fh.write(f"MODEL     {model_idx:>4d}\n")
    atom_idx = 1
    for c in range(n_chains):
        chain_id = chr(ord("A") + (c % 26))
        for i in range(N):
            rx, ry, rz = float(ca[c, i, 0]), float(ca[c, i, 1]), float(ca[c, i, 2])
            fh.write(
                f"ATOM  {atom_idx:5d}  CA  GLY {chain_id}{(i+1):4d}    "
                f"{rx:8.3f}{ry:8.3f}{rz:8.3f}  1.00  0.00           C\n"
            )
            atom_idx += 1
            sx, sy, sz = float(sg[c, i, 0]), float(sg[c, i, 1]), float(sg[c, i, 2])
            fh.write(
                f"HETATM{atom_idx:5d}  SG  GLY {chain_id}{(i+1):4d}    "
                f"{sx:8.3f}{sy:8.3f}{sz:8.3f}  1.00  0.00           S\n"
            )
            atom_idx += 1
        fh.write("TER\n")
    fh.write("ENDMDL\n")


def pdb_close_trajectory(fh) -> None:
    """Write END record and close the trajectory handle."""
    fh.write("END\n")
    fh.close()


def write_pdb_snapshot(path: str, ca, sg, model_idx: int = 1) -> None:
    """Single-snapshot PDB writer (legacy compatibility for one-shot dumps).

    Equivalent to opening a trajectory, writing one MODEL, and closing.
    """
    fh = pdb_open_trajectory(path)
    pdb_write_model(fh, ca, sg, model_idx)
    pdb_close_trajectory(fh)
