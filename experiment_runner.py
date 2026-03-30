"""
Comprehensive experiment runner for Rouse model MC simulation.
Tests: multistep MC batch sizes, GPU vs CPU, CA contact energy.
"""

import sys, os, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouse_model_python.config import SimulationConfig, CHAIN_CONFIGS
from rouse_model_python.chain import ChainState
from rouse_model_python.number_space import NumberSpace
from rouse_model_python.observables import StaticObservables
from rouse_model_python.simulation import RouseSimulation, SimulationStats


def check_bonds(pos_np_or_tensor, ns, l0):
    """Return (n_broken, max_bl, min_bl)."""
    if isinstance(pos_np_or_tensor, np.ndarray):
        pt = torch.from_numpy(pos_np_or_tensor)
    else:
        pt = pos_np_or_tensor
    bonds = ns.mic_delta(pt[:, :-1, :], pt[:, 1:, :])
    bl = torch.sqrt((bonds * bonds).sum(dim=-1))
    bad = ((bl - l0).abs() > 0.01).sum().item()
    return bad, bl.max().item(), bl.min().item()


def run_fast_experiment(N, n_chains, eq_sweeps, prod_sweeps, move_size,
                        contact_energy=0.0, box_size=None, seed=42):
    """Run experiment with numba/CPU fast path."""
    import rouse_model_python.fast_sweep as fs
    from rouse_model_python.fast_energy import FastEnergyComputer
    from rouse_model_python.fast_simulation import FastRouseSimulation

    # Override MOVE_SIZE
    old_ms = fs.MOVE_SIZE
    fs.MOVE_SIZE = move_size

    # Build config
    if box_size is None:
        box_size = CHAIN_CONFIGS.get(N, (n_chains, eq_sweeps, prod_sweeps, 300.0))[3]
    cfg = SimulationConfig(
        N=N, n_chains=n_chains, eq_sweeps=eq_sweeps, prod_sweeps=prod_sweeps,
        box_size=box_size, seed=seed, device="cpu",
        contact_energy=contact_energy,
    )

    ns = NumberSpace.from_config(cfg)
    state = ChainState(cfg)
    gen = cfg.get_torch_gen()
    state.initialize_random_walk(gen)
    state.wrap_all()
    pos_np = state.positions.cpu().numpy().copy()

    energy_comp = FastEnergyComputer(cfg)
    stats = SimulationStats()
    rng = np.random.RandomState(seed)

    # Equilibration
    t0 = time.time()
    for sweep in range(eq_sweeps):
        fs.fast_perform_sweep(pos_np, state.segments, energy_comp, cfg, stats, rng)
    eq_time = time.time() - t0

    # Check bonds after eq
    bad_eq, mx_eq, mn_eq = check_bonds(pos_np, ns, cfg.l0)

    # Production
    t0 = time.time()
    for sweep in range(prod_sweeps):
        fs.fast_perform_sweep(pos_np, state.segments, energy_comp, cfg, stats, rng)
    prod_time = time.time() - t0

    # Final observables
    state.positions.copy_(torch.from_numpy(pos_np))
    R2 = StaticObservables.compute_R2(state.positions, ns).mean().item()
    Rg2 = StaticObservables.compute_Rg2(state.positions, ns).mean().item()
    bad_final, mx_final, mn_final = check_bonds(pos_np, ns, cfg.l0)

    fs.MOVE_SIZE = old_ms  # restore

    return {
        "backend": "cpu_numba",
        "N": N, "n_chains": n_chains,
        "eq_sweeps": eq_sweeps, "prod_sweeps": prod_sweeps,
        "move_size": move_size,
        "contact_energy": contact_energy,
        "eq_time_s": round(eq_time, 1),
        "prod_time_s": round(prod_time, 1),
        "eq_rate_sw_s": round(eq_sweeps / eq_time, 2) if eq_time > 0 else 0,
        "prod_rate_sw_s": round(prod_sweeps / prod_time, 2) if prod_time > 0 else 0,
        "bonds_broken": bad_final,
        "max_bond_len": round(mx_final, 4),
        "min_bond_len": round(mn_final, 4),
        "R2": round(R2, 1),
        "Rg2": round(Rg2, 1),
        "R2_Rg2_ratio": round(R2 / Rg2, 2) if Rg2 > 0 else 0,
        "hinge_accept": f"{100*stats.hinge_accepted/max(1,stats.hinge_attempted):.1f}%",
        "tail_accept": f"{100*stats.tail_accepted/max(1,stats.tail_attempted):.1f}%",
        "pivot_accept": f"{100*stats.pivot_accepted/max(1,stats.pivot_attempted):.1f}%",
    }


def run_gpu_experiment(N, n_chains, eq_sweeps, prod_sweeps, move_size,
                       contact_energy=0.0, box_size=None, seed=42):
    """Run experiment with PyTorch GPU path."""
    from rouse_model_python.multistep_mc import perform_sweep, MOVE_SIZE as MC_MOVE_SIZE
    import rouse_model_python.multistep_mc as mc
    from rouse_model_python.energy import EnergyComputer

    old_ms = mc.MOVE_SIZE
    mc.MOVE_SIZE = move_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if box_size is None:
        box_size = CHAIN_CONFIGS.get(N, (n_chains, eq_sweeps, prod_sweeps, 300.0))[3]

    cfg = SimulationConfig(
        N=N, n_chains=n_chains, eq_sweeps=eq_sweeps, prod_sweeps=prod_sweeps,
        box_size=box_size, seed=seed, device=device,
        contact_energy=contact_energy,
        use_batched_mode=(device == "cuda"),
    )

    ns = NumberSpace.from_config(cfg)
    state = ChainState(cfg)
    gen = cfg.get_torch_gen()
    energy_comp = EnergyComputer(cfg, ns)
    stats = SimulationStats()

    state.initialize_random_walk(gen)
    state.wrap_all()

    # Equilibration
    t0 = time.time()
    for sweep in range(eq_sweeps):
        perform_sweep(state, energy_comp, gen, cfg, stats)
    eq_time = time.time() - t0

    bad_eq, mx_eq, mn_eq = check_bonds(state.positions.cpu(), ns, cfg.l0)

    # Production
    t0 = time.time()
    for sweep in range(prod_sweeps):
        perform_sweep(state, energy_comp, gen, cfg, stats)
    prod_time = time.time() - t0

    # Final observables
    pos_cpu = state.positions.cpu()
    ns_cpu = NumberSpace(ns.box_size, ns.sigma, device=torch.device('cpu'))
    R2 = StaticObservables.compute_R2(pos_cpu, ns_cpu).mean().item()
    Rg2 = StaticObservables.compute_Rg2(pos_cpu, ns_cpu).mean().item()
    bad_final, mx_final, mn_final = check_bonds(pos_cpu, ns_cpu, cfg.l0)

    mc.MOVE_SIZE = old_ms

    return {
        "backend": f"gpu_{device}" if device == "cuda" else "cpu_pytorch",
        "N": N, "n_chains": n_chains,
        "eq_sweeps": eq_sweeps, "prod_sweeps": prod_sweeps,
        "move_size": move_size,
        "contact_energy": contact_energy,
        "eq_time_s": round(eq_time, 1),
        "prod_time_s": round(prod_time, 1),
        "eq_rate_sw_s": round(eq_sweeps / eq_time, 2) if eq_time > 0 else 0,
        "prod_rate_sw_s": round(prod_sweeps / prod_time, 2) if prod_time > 0 else 0,
        "bonds_broken": bad_final,
        "max_bond_len": round(mx_final, 4),
        "min_bond_len": round(mn_final, 4),
        "R2": round(R2, 1),
        "Rg2": round(Rg2, 1),
        "R2_Rg2_ratio": round(R2 / Rg2, 2) if Rg2 > 0 else 0,
        "hinge_accept": f"{100*stats.hinge_accepted/max(1,stats.hinge_attempted):.1f}%",
        "tail_accept": f"{100*stats.tail_accepted/max(1,stats.tail_attempted):.1f}%",
        "pivot_accept": f"{100*stats.pivot_accepted/max(1,stats.pivot_attempted):.1f}%",
    }


def print_result(r, label=""):
    """Pretty-print one experiment result."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  Backend:        {r['backend']}")
    print(f"  N={r['N']}, chains={r['n_chains']}, MOVE_SIZE={r['move_size']}")
    print(f"  Contact energy: {r['contact_energy']} kJ/mol")
    print(f"  Eq:  {r['eq_sweeps']} sweeps in {r['eq_time_s']}s ({r['eq_rate_sw_s']} sw/s)")
    print(f"  Prod: {r['prod_sweeps']} sweeps in {r['prod_time_s']}s ({r['prod_rate_sw_s']} sw/s)")
    print(f"  Bonds broken: {r['bonds_broken']}  (max={r['max_bond_len']}, min={r['min_bond_len']})")
    print(f"  R²={r['R2']}, Rg²={r['Rg2']}, R²/Rg²={r['R2_Rg2_ratio']}")
    print(f"  Accept: hinge={r['hinge_accept']}, tail={r['tail_accept']}, pivot={r['pivot_accept']}")


if __name__ == "__main__":
    results = []
    # Test params: N=50, 30 chains, 500+500 sweeps for fast iteration
    N = 50; NC = 30; EQ = 500; PROD = 500; BOX = 238.0

    print("=" * 70)
    print("ROUSE MODEL MC — COMPREHENSIVE EXPERIMENTS")
    print("=" * 70)

    # ── Experiment A: MOVE_SIZE sweep (CPU/numba) ──────────────────
    print("\n\n### EXPERIMENT A: MOVE_SIZE sweep (CPU/numba) ###")
    for ms in [1, 5, 10, 20]:
        r = run_fast_experiment(N, NC, EQ, PROD, move_size=ms, box_size=BOX)
        results.append(r)
        print_result(r, f"MOVE_SIZE={ms} (CPU/numba)")

    # ── Experiment B: GPU vs CPU ──────────────────────────────────
    print("\n\n### EXPERIMENT B: GPU vs CPU ###")
    # GPU batched
    for ms in [1, 10, 20]:
        try:
            r = run_gpu_experiment(N, NC, EQ, PROD, move_size=ms, box_size=BOX)
            results.append(r)
            print_result(r, f"MOVE_SIZE={ms} (GPU/PyTorch)")
        except Exception as e:
            print(f"  GPU MOVE_SIZE={ms} FAILED: {e}")
            results.append({"backend": "gpu_cuda", "move_size": ms, "error": str(e)})

    # ── Experiment C: CA Contact Energy ───────────────────────────
    print("\n\n### EXPERIMENT C: CA Contact Energy ###")
    for ce in [0.0, -0.5, -1.0, -2.0]:
        r = run_fast_experiment(N, NC, EQ, PROD, move_size=20,
                                contact_energy=ce, box_size=BOX)
        results.append(r)
        print_result(r, f"CONTACT_ENERGY={ce} kJ/mol")

    # ── Experiment D: Larger system (N=250) ───────────────────────
    print("\n\n### EXPERIMENT D: Larger system scaling ###")
    for backend in ["cpu", "gpu"]:
        try:
            if backend == "cpu":
                r = run_fast_experiment(250, 10, 200, 200, move_size=20, box_size=293.0)
            else:
                r = run_gpu_experiment(250, 10, 200, 200, move_size=20, box_size=293.0)
            results.append(r)
            print_result(r, f"N=250 ({backend})")
        except Exception as e:
            print(f"  N=250 {backend} FAILED: {e}")
            results.append({"backend": backend, "N": 250, "error": str(e)})

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "rouse_python_validation_deliverable_2016_MAR_26",
                             "experiment_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to {out_path}")
