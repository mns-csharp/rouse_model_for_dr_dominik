"""Simulation driver — torch CPU conventional MC, single-thread."""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

import logging

from . import io_utils, mc, observables
logger = logging.getLogger(__name__)
from .chain import ChainState
from .config import SimConfig
from .scratch import GpuScratch


class Simulation:

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.state = ChainState(cfg)
        self.rng = np.random.default_rng(cfg.seed)
        self.stats = mc.SimulationStats()
        self.scratch = GpuScratch(
            self.state,
            total_segments=self.state.segments.total_segments,
            M=max(int(self.state.segments.table[:, 2].max()), 1),
            B=int(cfg.n_chains),
        )

    def initialize(self) -> None:
        self.state.initialize(self.rng)
        self.state.rescale_bonds()

    def _maybe_rescale(self, sweep_idx: int) -> None:
        cfg = self.cfg
        if cfg.n_small_steps > 0 and (sweep_idx + 1) % cfg.n_small_steps == 0:
            drift, pre = self.state.rescale_bonds()
            rs = getattr(cfg, "_rescale_state", None)
            if rs is None:
                rs = {"count": 0, "max_drift": 0.0, "last_drift": 0.0}
                cfg._rescale_state = rs
            rs["count"] += 1
            rs["last_drift"] = float(drift)
            if drift > rs["max_drift"]:
                rs["max_drift"] = float(drift)
            ss = getattr(cfg, "_stretch_state", None)
            if ss is None:
                ss = {"max": 0.0}
                cfg._stretch_state = ss
            if pre > ss["max"]:
                ss["max"] = float(pre)
            logger.info('bond rescale @ sweep %d: max_stretch_pre=%.5f', sweep_idx + 1, pre)
            if pre > self.cfg.bond_stretch_warn:
                logger.warning('bond stretch %.4f exceeds warn %.4f', pre, self.cfg.bond_stretch_warn)
            if pre > cfg.bond_stretch_raise:
                raise RuntimeError(f"bond stretch {pre:.4f} > raise threshold")

    def _effective_traj_stride(self) -> int:
        cfg = self.cfg
        s = int(getattr(cfg, "traj_stride", 0) or 0)
        if s <= 0:
            s = max(1, int(getattr(cfg, "sample_interval", 1)))
        return s

    def _ca_sg_numpy(self):
        ca = self.state.ca
        sg = self.state.sg
        if hasattr(ca, "detach"):
            ca = ca.detach().cpu().numpy()
            sg = sg.detach().cpu().numpy()
        return ca, sg

    def _run_phase(self, phase: str, n_sweeps: int,
                   log_path: Optional[str], traj_path: Optional[str]) -> dict:
        cfg = self.cfg
        rows = []
        sweep_walls_ms = np.zeros(n_sweeps, dtype=np.float64)
        kernel_walls_ms = np.zeros(n_sweeps, dtype=np.float64)
        hb = int(getattr(cfg, "_heartbeat_interval", 50) or 0)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info("%s start: %s_sweeps=%d", phase, phase, n_sweeps)
        traj_stride = self._effective_traj_stride()
        traj_fh = io_utils.pdb_open_trajectory(
            traj_path, remark=f"{phase} trajectory; sample_stride={traj_stride}"
        ) if traj_path else None
        snap = 0
        t0 = time.perf_counter()
        t_last = t0
        for sw in range(n_sweeps):
            t_sw = time.perf_counter()
            kernel_ms = mc.perform_sweep(self.state, cfg, self.stats, self.rng, self.scratch)
            sweep_walls_ms[sw] = (time.perf_counter() - t_sw) * 1000.0
            if kernel_ms is not None:
                kernel_walls_ms[sw] = float(kernel_ms)
            self._maybe_rescale(sw)
            if (sw + 1) % max(1, cfg.sample_interval) == 0:
                obs = observables.summary(self.state)
                e_total = observables.total_energy(self.state, cfg)
                rows.append([sw + 1, obs["rg2_mean"], obs["re2_mean"], e_total,
                             self.stats.acceptance("hinge"),
                             self.stats.acceptance("n_tail"),
                             self.stats.acceptance("c_tail")])
            if traj_fh and (sw + 1) % traj_stride == 0:
                snap += 1
                ca_np, sg_np = self._ca_sg_numpy()
                io_utils.pdb_write_model(traj_fh, ca_np, sg_np, model_idx=snap)
            if hb > 0 and (sw + 1) % hb == 0:
                now = time.perf_counter()
                avg_ms = float(sweep_walls_ms[sw + 1 - hb : sw + 1].mean())
                logger.info(
                    "%s sweep %d/%d (%.1f%%) avg_sweep=%.2fms last_%d_wall=%.2fs "
                    "acc_h=%.3f acc_n=%.3f acc_c=%.3f",
                    phase, sw + 1, n_sweeps, 100.0 * (sw + 1) / n_sweeps,
                    avg_ms, hb, now - t_last,
                    self.stats.acceptance("hinge"),
                    self.stats.acceptance("n_tail"),
                    self.stats.acceptance("c_tail"),
                )
                t_last = now
        wall = time.perf_counter() - t0
        peak_mb = (torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                   if torch.cuda.is_available() else 0.0)
        if traj_fh:
            io_utils.pdb_close_trajectory(traj_fh)
        if log_path:
            io_utils.write_tsv(log_path,
                ["sweep", "Rg2", "Re2", "E_total",
                 "acc_hinge", "acc_ntail", "acc_ctail"], rows)
            walls_path = os.path.join(os.path.dirname(log_path) or ".",
                                      f"{phase}_sweep_walls.tsv")
            io_utils.write_sweep_walls_tsv(walls_path, sweep_walls_ms, kernel_walls_ms)
        return {"wall_s": wall, "n_sweeps": n_sweeps, "rows": rows,
                "sweep_walls_ms": sweep_walls_ms, "kernel_walls_ms": kernel_walls_ms,
                "peak_mb": peak_mb}

    def run_equilibration(self, log_path: Optional[str] = None,
                          traj_path: Optional[str] = None) -> dict:
        return self._run_phase("eq", self.cfg.eq_sweeps, log_path, traj_path)

    def run_production(self, log_path: Optional[str] = None,
                       traj_path: Optional[str] = None) -> dict:
        self.stats = mc.SimulationStats()
        return self._run_phase("prod", self.cfg.prod_sweeps, log_path, traj_path)

    def write_summary(self, path: str, eq, prod) -> None:
        cfg = self.cfg
        obs = observables.summary(self.state)
        payload = {
            "config": {
                "N": cfg.N, "n_chains": cfg.n_chains, "phi": cfg.phi,
                "box_size": cfg.box_size, "sigma": cfg.sigma, "l0": cfg.l0,
                "kBT": cfg.kBT,
                "max_angle_hinge": cfg.max_angle_hinge,
                "residues_per_segment": cfg.residues_per_segment,
                "seed": cfg.seed,
                "init_method": cfg.init_method,
                "eq_sweeps": cfg.eq_sweeps, "prod_sweeps": cfg.prod_sweeps,
                "algorithm": "segmented_conventional_cuda_c_cell_list",
                "device": "gpu",
                "thread_mode": "single_thread",
                "backend": "cuda_c",
                "cuda_device": cfg.gpu_device,
            },
            "observables": obs,
            "acceptance": {
                "hinge": self.stats.acceptance("hinge"),
                "n_tail": self.stats.acceptance("n_tail"),
                "c_tail": self.stats.acceptance("c_tail"),
            },
            "timing": {
                "eq_wall_s": eq["wall_s"],
                "prod_wall_s": prod["wall_s"],
                "eq_sweeps_per_s": eq["n_sweeps"] / max(eq["wall_s"], 1e-12),
                "prod_sweeps_per_s": prod["n_sweeps"] / max(prod["wall_s"], 1e-12),
                "eq_sweep_ms": _stats_ms(eq.get("sweep_walls_ms")),
                "prod_sweep_ms": _stats_ms(prod.get("sweep_walls_ms")),
                "eq_kernel_ms": _stats_ms(eq.get("kernel_walls_ms")),
                "prod_kernel_ms": _stats_ms(prod.get("kernel_walls_ms")),
            },
            "gpu": _gpu_block(cfg, eq, prod),
        }
        io_utils.write_json(path, payload)


def _stats_ms(arr) -> dict:
    if arr is None or len(arr) == 0:
        return {}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
    }


def _gpu_block(cfg, eq, prod) -> dict:
    if not torch.cuda.is_available():
        return {}
    dev = getattr(cfg, "gpu_device", 0)
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "device_name": torch.cuda.get_device_name(dev),
        "peak_mb_eq": float(eq.get("peak_mb", 0.0)),
        "peak_mb_prod": float(prod.get("peak_mb", 0.0)),
        "free_mb_at_summary": free_b / (1024.0 * 1024.0),
        "total_mb": total_b / (1024.0 * 1024.0),
    }
