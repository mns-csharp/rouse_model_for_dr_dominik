"""Numerical-equivalence check for the g1m_ccx streamed CUDA-C optimizations.

Compares the optimized CUDA-C hot-path against the original (correct) torch
fused path on a fixed-seed real state, batch by batch. Athermal hard-sphere
(contact_energy=0, repulsive_energy=1e6) -> delta-E is integer multiples of
rep_e, so equivalence should be near-bit-exact.

Run from repo root:  python _test_ccx_equivalence.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

_APP = "g1m_ccx_gpu_single_thread_multistep_mc_cuda_c_streamed"
from importlib import import_module

config_mod = import_module(f"{_APP}.config")
sim_mod = import_module(f"{_APP}.simulation")
mc_mod = import_module(f"{_APP}.mc")
proposer_mod = import_module(f"{_APP}.proposer")
energy_mod = import_module(f"{_APP}.energy")

SimConfig = config_mod.SimConfig
Simulation = sim_mod.Simulation
propose_batch_torch = proposer_mod.propose_batch_torch
batch_delta_e_torch_fused = energy_mod.batch_delta_e_torch_fused
batch_delta_e_cuda = energy_mod.batch_delta_e_cuda
correction_matrix_torch_fused = energy_mod.correction_matrix_torch_fused
correction_matrix_cuda = energy_mod.correction_matrix_cuda


def main() -> int:
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available — equivalence test requires a GPU.")
        return 0
    torch.cuda.set_device(0)

    cfg = SimConfig(
        N=50, n_chains=64, phi=0.10,
        eq_sweeps=2, prod_sweeps=2, seed=42,
        residues_per_segment=8, batch_size=256,
        max_angle_hinge=0.5 * math.pi,
        init_method="random_saw",
        contact_energy=0.0, repulsive_energy=1e6,
        output_dir="./_ccx_eqtest",
    )
    sim = Simulation(cfg)
    sim.initialize()
    state = sim.state
    rng = sim.rng

    seg_info = state.segments
    N = cfg.N
    box = cfg.box_size
    M = max(int(seg_info.table[:, 2].max()), 1)
    n_chains = cfg.n_chains
    device = state.ca.device
    dtype = state.ca.dtype
    ca_t = state.ca
    sg_t = state.sg
    table_full = seg_info.table

    perm, boundaries = mc_mod._build_round_permutation(seg_info, n_chains, rng)
    B_max = min(int(cfg.batch_size), n_chains)
    batch_ranges = []
    for ri in range(len(boundaries) - 1):
        for bs in range(boundaries[ri], boundaries[ri + 1], B_max):
            be = min(bs + B_max, boundaries[ri + 1])
            batch_ranges.append((bs, be))

    de_max_abs = 0.0
    de_mismatch = 0
    corr_max_abs = 0.0
    corr_mismatch = 0
    n_checked = 0
    for bs, be in batch_ranges:
        B = be - bs
        if B == 0:
            continue
        meta_np = table_full[perm[bs:be]]
        meta_t = torch.from_numpy(meta_np).to(device)
        angle_unif = torch.rand(B, device=device, dtype=dtype)

        old_ca, new_ca, old_sg, new_sg, n_moved_out, move_type = propose_batch_torch(
            ca_t, sg_t, meta_t, angle_unif, M, N, box, cfg.max_angle_hinge,
        )
        ci_l = meta_t[:, 0].long()
        bs_l = meta_t[:, 1].long()
        de_args = (
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            ci_l, bs_l, ca_t, sg_t, N, box,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )
        de_torch = batch_delta_e_torch_fused(*de_args)
        de_cuda = batch_delta_e_cuda(*de_args)

        corr_args = (
            old_ca, new_ca, old_sg, new_sg, n_moved_out,
            ci_l, bs_l, N, box,
            cfg.r_rep_sq, cfg.r_max_sq, cfg.repulsive_energy, cfg.contact_energy,
            cfg.min_seq_caca, cfg.min_seq_casg, cfg.min_seq_sgsg,
        )
        corr_torch = correction_matrix_torch_fused(*corr_args)
        corr_cuda = correction_matrix_cuda(*corr_args)
        torch.cuda.synchronize()

        de_diff = (de_torch.detach().double() - de_cuda.detach().double()).abs()
        de_max_abs = max(de_max_abs, float(de_diff.max().item()))
        de_mismatch += int((de_diff > 1.0).sum().item())

        # CUDA correction is upper-triangular; compare only j>i entries.
        tri = torch.triu(torch.ones(B, B, device=device, dtype=torch.bool), diagonal=1)
        cd = (corr_torch.detach().double() - corr_cuda.detach().double()).abs()
        cd = cd * tri.double()
        corr_max_abs = max(corr_max_abs, float(cd.max().item()))
        corr_mismatch += int((cd > 1.0).sum().item())
        n_checked += B

    print(f"batches={len(batch_ranges)}  proposals checked={n_checked}")
    print(f"delta-E:    max |torch - cuda| = {de_max_abs:.6g}   mismatches(>1.0) = {de_mismatch}")
    print(f"correction: max |torch - cuda| = {corr_max_abs:.6g}   mismatches(>1.0) = {corr_mismatch}")
    if de_mismatch == 0 and corr_mismatch == 0:
        print("[PASS] streamed CUDA-C delta-E + correction match the torch reference.")
        return 0
    print("[FAIL] streamed CUDA-C path diverges from the torch reference.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
