"""
Scaling benchmark: measure simulation time vs number of beads (N)
for 4 device/batching configurations.

N = 25, 500, 1000  |  100 chains  |  10000 sweeps (5000 eq + 5000 prod)
Configs:
  1. GPU + batched
  2. CPU + batched
  3. GPU + unbatched
  4. CPU + unbatched

Saves results incrementally to scaling_benchmark_results.json after each run.
Supports --resume to skip already-completed runs.
"""

import sys
import os
import time
import json
import random
import math
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouse_model_python.config import SimulationConfig, SIGMA, SEED
from rouse_model_python.simulation import RouseSimulation

EQ_SWEEPS = 5000
PROD_SWEEPS = 5000
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaling_benchmark_results.json")


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(N, n_chains, device, batched):
    phi = 0.01
    sigma3 = SIGMA ** 3
    box_size = (n_chains * N * sigma3 / phi) ** (1.0 / 3.0)
    rms_R = SIGMA * (N ** 0.588)
    box_size = max(box_size, 3.0 * rms_R)

    cfg = SimulationConfig(
        N=N,
        n_chains=n_chains,
        eq_sweeps=EQ_SWEEPS,
        prod_sweeps=PROD_SWEEPS,
        box_size=box_size,
        seed=SEED,
        device="cuda" if device == "gpu" else "cpu",
        use_batched_mode=batched,
        target_phi=phi,
    )
    return cfg


def run_single(N, n_chains, device, batched):
    set_seeds()
    cfg = make_config(N, n_chains, device, batched)

    print(f"\n  N={N}, chains={n_chains}, device={device}, batched={batched}, "
          f"sweeps={EQ_SWEEPS+PROD_SWEEPS}, box={cfg.box_size:.1f}A, "
          f"beads={cfg.total_beads}", flush=True)

    if device == "gpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    sim = RouseSimulation(cfg)
    results = sim.run()

    if device == "gpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    print(f"  -> Wall time: {elapsed:.2f}s ({elapsed/3600:.2f}h)", flush=True)
    return elapsed


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"parameters": {}, "timings": {}}


def save_results(data):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def print_summary(all_timings, N_values, n_chains, configs):
    print("\n\n" + "=" * 90)
    print("SCALING BENCHMARK RESULTS")
    print(f"({n_chains} chains, {EQ_SWEEPS+PROD_SWEEPS} sweeps)")
    print("=" * 90)

    header = f"{'Configuration':<20}"
    for N in N_values:
        header += f" | {'N='+str(N):>12}"
    header += f" | {'1000/25':>10}"
    print(header)
    print("-" * 90)

    for device, batched, label in configs:
        if label not in all_timings:
            continue
        t = all_timings[label]
        if not all(str(n) in t for n in N_values):
            continue
        row = f"{label:<20}"
        for N in N_values:
            secs = t[str(N)]
            if secs < 60:
                row += f" | {secs:>10.1f}s "
            elif secs < 3600:
                row += f" | {secs/60:>10.1f}m "
            else:
                row += f" | {secs/3600:>10.2f}h "
        t25 = t.get("25", 0)
        t1000 = t.get("1000", 0)
        ratio = t1000 / t25 if t25 > 0 else float('nan')
        row += f" | {ratio:>9.1f}x"
        print(row)

    # Scaling analysis
    print("\n" + "=" * 90)
    print("SCALING ANALYSIS (wall time vs N)")
    print("=" * 90)
    for device, batched, label in configs:
        if label not in all_timings:
            continue
        t = all_timings[label]
        if not all(str(n) in t for n in N_values):
            continue
        ns = np.array(N_values, dtype=float)
        times = np.array([t[str(n)] for n in N_values])
        log_n = np.log(ns)
        log_t = np.log(times)
        alpha, log_c0 = np.polyfit(log_n, log_t, 1)
        c0 = math.exp(log_c0)
        print(f"  {label:<20}: time ~ {c0:.4f} * N^{alpha:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip already-completed runs")
    parser.add_argument("--summary", action="store_true", help="Just print summary from saved results")
    args = parser.parse_args()

    N_values = [25, 500, 1000]
    n_chains = 100

    configs = [
        ("gpu", True,  "GPU+batched"),
        ("cpu", True,  "CPU+batched"),
        ("gpu", False, "GPU+unbatched"),
        ("cpu", False, "CPU+unbatched"),
    ]

    saved = load_results()

    if args.summary:
        print_summary(saved.get("timings", {}), N_values, n_chains, configs)
        return

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU available. Skipping GPU configurations.")
        configs = [(d, b, l) for d, b, l in configs if d != "gpu"]

    print(f"CPU cores: {os.cpu_count()}")
    print(f"N values: {N_values}, chains: {n_chains}, sweeps: {EQ_SWEEPS+PROD_SWEEPS}")

    saved["parameters"] = {
        "N_values": N_values,
        "n_chains": n_chains,
        "eq_sweeps": EQ_SWEEPS,
        "prod_sweeps": PROD_SWEEPS,
        "total_sweeps": EQ_SWEEPS + PROD_SWEEPS,
    }

    for device, batched, label in configs:
        print(f"\n{'='*70}")
        print(f"  Configuration: {label}")
        print(f"{'='*70}")

        if label not in saved["timings"]:
            saved["timings"][label] = {}

        for N in N_values:
            key = str(N)
            if args.resume and key in saved["timings"][label]:
                t = saved["timings"][label][key]
                print(f"  N={N}: already done ({t:.2f}s), skipping", flush=True)
                continue

            elapsed = run_single(N, n_chains, device, batched)
            saved["timings"][label][key] = elapsed
            save_results(saved)
            print(f"  [saved to {RESULTS_FILE}]", flush=True)

    print_summary(saved["timings"], N_values, n_chains, configs)
    save_results(saved)
    print(f"\nFinal results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
