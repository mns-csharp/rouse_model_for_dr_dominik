"""
Benchmark: profile perform_sweep for N=25 and N=250 across modes.
Identifies bottlenecks by timing individual components.

Usage:
    python -m rouse_model_python.benchmark --device {cpu|gpu|mixed} --is_parallel {true|false}

Both --device and --is_parallel are REQUIRED.
"""

import sys, os, time, random, math
import numpy as np
import torch
from functools import wraps
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouse_model_python.config import SimulationConfig, SEED
from rouse_model_python.chain import ChainState
from rouse_model_python.energy import EnergyComputer
from rouse_model_python.number_space import NumberSpace
from rouse_model_python.multistep_mc import perform_sweep, MOVE_SIZE
from rouse_model_python.simulation import SimulationStats
from rouse_model_python.execution_policy import parse_execution_args

# ---------------------------------------------------------------------------
# Monkey-patch timing instrumentation
# ---------------------------------------------------------------------------

TIMINGS = defaultdict(lambda: {"calls": 0, "total": 0.0})

def timed(label):
    """Decorator that accumulates wall-clock time under `label`."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            TIMINGS[label]["calls"] += 1
            TIMINGS[label]["total"] += dt
            return result
        return wrapper
    return decorator


def instrument():
    """Patch key functions with timing wrappers."""
    import rouse_model_python.multistep_mc as mmc
    import rouse_model_python.mc_moves as mcm
    import rouse_model_python.energy as eng

    # Sweep-level
    mmc._process_batch = timed("_process_batch")(mmc._process_batch)

    # Move proposals
    mcm.propose_hinge_move = timed("propose_hinge")(mcm.propose_hinge_move)
    mcm.propose_tail_move = timed("propose_tail")(mcm.propose_tail_move)
    mcm.propose_pivot_move = timed("propose_pivot")(mcm.propose_pivot_move)
    mcm.propose_pivot_move_pooled = timed("propose_pivot_pooled")(mcm.propose_pivot_move_pooled)
    mcm.propose_batch_segment_moves = timed("propose_batch_seg")(mcm.propose_batch_segment_moves)
    mcm.propose_segment_move = timed("propose_segment")(mcm.propose_segment_move)

    # Energy
    eng.EnergyComputer.rebuild_cell_list = timed("rebuild_cell_list")(eng.EnergyComputer.rebuild_cell_list)
    eng.EnergyComputer.compute_delta_energy_move = timed("delta_energy_move")(eng.EnergyComputer.compute_delta_energy_move)
    eng.EnergyComputer.compute_batch_energy_matrices = timed("batch_energy_matrices")(eng.EnergyComputer.compute_batch_energy_matrices)
    eng.EnergyComputer.compute_batch_delta_energy = timed("batch_delta_energy")(eng.EnergyComputer.compute_batch_delta_energy)

    # Cell list
    eng.CellList.build = timed("cell_list_build")(eng.CellList.build)
    eng.CellList.gather_neighbors = timed("gather_neighbors")(eng.CellList.gather_neighbors)

    # Rotation helpers
    mcm.rodrigues_rotation_matrix = timed("rodrigues")(mcm.rodrigues_rotation_matrix)
    mcm.random_so3_matrix = timed("random_so3")(mcm.random_so3_matrix)
    mcm.apply_rotation_to_beads_unwrapped = timed("apply_rotation")(mcm.apply_rotation_to_beads_unwrapped)
    mcm._batched_rodrigues = timed("batched_rodrigues")(mcm._batched_rodrigues)


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def run_benchmark(N, device, batched, n_sweeps=5):
    """Run n_sweeps and return per-sweep time + component breakdown.

    Args:
        N: chain length
        device: torch device string -- explicitly provided, no default
        batched: whether to use batched energy computation
        n_sweeps: number of sweeps to benchmark
    """
    TIMINGS.clear()
    set_seeds()

    cfg = SimulationConfig.for_chain_length(N, device=device)
    cfg.use_batched_mode = batched

    state = ChainState(cfg)
    ns = NumberSpace.from_config(cfg)
    energy_comp = EnergyComputer(cfg, ns)
    gen = cfg.get_torch_gen()
    stats = SimulationStats()

    # Initialize
    state.initialize_random_walk(gen)
    state.wrap_all()

    # Warm-up: 1 sweep (not timed at component level, but warms caches)
    TIMINGS.clear()
    perform_sweep(state, energy_comp, gen, cfg, stats)
    TIMINGS.clear()

    # Timed sweeps
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_sweeps):
        perform_sweep(state, energy_comp, gen, cfg, stats)
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    return wall, n_sweeps, dict(TIMINGS)


def fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds*1e6:8.0f} us"
    if seconds < 1.0:
        return f"{seconds*1e3:8.1f} ms"
    return f"{seconds:8.2f}  s"


def print_report(N, device, batched, wall, n_sweeps, timings):
    mode = "batched" if batched else "sequential"
    print(f"\n{'='*72}")
    print(f"  N={N}  device={device}  mode={mode}  sweeps={n_sweeps}")
    print(f"  Total wall: {wall:.3f}s  ({wall/n_sweeps:.3f}s/sweep)")
    print(f"{'='*72}")

    # Sort by total time descending
    items = sorted(timings.items(), key=lambda x: -x[1]["total"])
    total_accounted = sum(v["total"] for v in timings.values())

    print(f"  {'Component':<28s} {'Total':>10s} {'Calls':>8s} {'Per-call':>10s} {'% wall':>7s}")
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*10} {'-'*7}")
    for label, v in items:
        if v["total"] < 1e-6:
            continue
        per_call = v["total"] / max(v["calls"], 1)
        pct = 100.0 * v["total"] / wall if wall > 0 else 0
        print(f"  {label:<28s} {fmt_time(v['total']):>10s} {v['calls']:>8d} {fmt_time(per_call):>10s} {pct:>6.1f}%")

    unaccounted = wall - total_accounted
    if unaccounted > 0.001:
        pct = 100.0 * unaccounted / wall
        print(f"  {'(overhead/uninstrumented)':<28s} {fmt_time(unaccounted):>10s} {'':>8s} {'':>10s} {pct:>6.1f}%")


def main():
    policy, caps, cli_args = parse_execution_args()

    instrument()

    # Determine which device configs to benchmark based on policy
    torch_dev = policy.torch_device
    devices_to_test = [torch_dev]
    if policy.device == "mixed" and caps.n_gpus > 0:
        devices_to_test = ["cpu", "cuda"]
    elif policy.device == "gpu":
        devices_to_test = ["cuda"]
    elif policy.device == "cpu":
        devices_to_test = ["cpu"]

    n_sweeps = 5
    batched_only = policy.use_batched_mode
    configs = []
    for N in [25, 50, 100, 250, 500]:
        for device in devices_to_test:
            if batched_only:
                configs.append((N, device, True))
            else:
                for batched in [False, True]:
                    configs.append((N, device, batched))

    results = []
    for N, device, batched in configs:
        mode = "batched" if batched else "sequential"
        print(f"\nRunning N={N} device={device} mode={mode} ...", flush=True)
        wall, ns, timings = run_benchmark(N, device, batched, n_sweeps)
        results.append((N, device, batched, wall, ns, timings))
        print_report(N, device, batched, wall, ns, timings)

    # Summary table
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'N':>5s} {'Device':>6s} {'Mode':<12s} {'Wall(s)':>8s} {'s/sweep':>8s} {'Top bottleneck':<30s} {'%':>5s}")
    print(f"  {'-'*5} {'-'*6} {'-'*12} {'-'*8} {'-'*8} {'-'*30} {'-'*5}")
    for N, device, batched, wall, ns, timings in results:
        mode = "batched" if batched else "sequential"
        per_sweep = wall / ns
        top = max(timings.items(), key=lambda x: x[1]["total"]) if timings else ("?", {"total": 0})
        top_pct = 100.0 * top[1]["total"] / wall if wall > 0 else 0
        print(f"  {N:>5d} {device:>6s} {mode:<12s} {wall:>8.2f} {per_sweep:>8.3f} {top[0]:<30s} {top_pct:>5.1f}")


if __name__ == "__main__":
    main()
