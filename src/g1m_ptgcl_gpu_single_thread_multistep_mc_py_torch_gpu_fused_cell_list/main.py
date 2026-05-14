"""CLI entry — single-thread GPU torch conventional MC, fused."""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

from . import io_utils
from . import _checklist
from . import mc as _mc_mod
from .logging_setup import configure_logging
from .config import SimConfig
from .simulation import Simulation


def parse_args(argv=None) -> SimConfig:
    p = argparse.ArgumentParser(
        description="gpu/single_thread/conventional_mc/py_torch_gpu_fused — GPU-resident fused MC",
    )
    p.add_argument("--N", type=int, default=25)
    p.add_argument("--n_chains", type=int, default=4)
    p.add_argument("--phi", type=float, default=0.10)
    p.add_argument("--eq_sweeps", type=int, default=100)
    p.add_argument("--prod_sweeps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--residues_per_segment", type=int, default=8)
    p.add_argument("--max_angle_hinge_pi", type=float, default=0.5)
    p.add_argument("--init_method", choices=["random_saw", "serpentine"],
                   default="random_saw")
    p.add_argument("--output_dir", default="./_smoke_app5b")
    p.add_argument("--traj_stride", type=int, default=0)
    p.add_argument("--contact_energy", type=float, default=0.0)
    p.add_argument("--repulsive_energy", type=float, default=1e6)
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log_file", default="",
                   help="if set, write logs to this file (default: <output_dir>/run.log)")
    p.add_argument("--heartbeat_interval", type=int, default=50,
                   help="emit INFO progress every N sweeps; 0 disables")
    p.add_argument("--cap_inner_hinge", action="store_true")
    p.add_argument("--cap_tail", action="store_true")
    p.add_argument("--batch_size", type=int, default=256,
                   help="multistep batch size B (host-side accept loop is O(B))")
    args = p.parse_args(argv)

    cfg = SimConfig(
        N=args.N, n_chains=args.n_chains, phi=args.phi,
        eq_sweeps=args.eq_sweeps, prod_sweeps=args.prod_sweeps,
        seed=args.seed,
        residues_per_segment=args.residues_per_segment,
        max_angle_hinge=args.max_angle_hinge_pi * math.pi,
        init_method=args.init_method,
        contact_energy=args.contact_energy,
        repulsive_energy=args.repulsive_energy,
        output_dir=args.output_dir,
    )
    cfg.traj_stride = args.traj_stride
    cfg._log_level = args.log_level
    cfg._log_file = args.log_file
    cfg._heartbeat_interval = args.heartbeat_interval
    cfg.cap_inner_hinge = args.cap_inner_hinge
    cfg.cap_tail = args.cap_tail
    cfg.batch_size = max(1, int(args.batch_size))
    return cfg


def main(argv=None) -> int:
    cfg = parse_args(argv)
    if not torch.cuda.is_available():
        print('[ERROR] CUDA not available; this app requires a GPU.', flush=True)
        return 1
    torch.cuda.set_device(cfg.gpu_device)

    print(f"[INDEP-gpu-single-multistep-pytorch-cl] N={cfg.N} K={cfg.n_chains} "
          f"phi={cfg.phi:.3f} eq={cfg.eq_sweeps} prod={cfg.prod_sweeps} "
          f"box={cfg.box_size:.2f}A", flush=True)

    out = io_utils.ensure_dir(cfg.output_dir)
    log_file = getattr(cfg, '_log_file', '') or os.path.join(out, 'run.log')
    log = configure_logging(getattr(cfg, '_log_level', 'INFO'), log_file, 'INDEP-gpu-single-multistep-pytorch-cl')

    app_name = "g1m_ptgcl_gpu_single_thread_multistep_mc_py_torch_gpu_fused_cell_list"
    _checklist.emit_static(cfg, app_name=app_name)

    _mc_mod._reset_counters()
    sim = Simulation(cfg)
    sim.initialize()
    _checklist.emit_post_init(sim.state, cfg)
    _checklist.emit_synthetic(cfg, sim.rng)

    eq_traj_path = os.path.join(out, "eq_trajectory.pdb")
    prod_traj_path = os.path.join(out, "prod_trajectory.pdb")
    traj_dir = out

    _checklist.emit_phase_boundary("eq")
    eq = sim.run_equilibration(log_path=os.path.join(out, "eq_observables.tsv"),
                               traj_path=eq_traj_path)
    _checklist.emit_post_eq(eq, cfg)
    print(f"[INDEP-gpu-single-multistep-pytorch-cl] eq done in {eq['wall_s']:.2f}s "
          f"({eq['n_sweeps']/max(eq['wall_s'],1e-12):.2f} sweeps/s)", flush=True)

    _checklist.emit_phase_boundary("prod")
    prod = sim.run_production(log_path=os.path.join(out, "prod_observables.tsv"),
                              traj_path=prod_traj_path)
    print(f"[INDEP-gpu-single-multistep-pytorch-cl] prod done in {prod['wall_s']:.2f}s "
          f"({prod['n_sweeps']/max(prod['wall_s'],1e-12):.2f} sweeps/s)", flush=True)

    summary_path = os.path.join(out, "summary.json")
    sim.write_summary(summary_path, eq, prod)

    _checklist.emit_post_run(prod, traj_dir, cfg, summary_path,
                             sim.stats, sim.state, eq, app_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
