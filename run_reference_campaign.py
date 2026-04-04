"""
Recreate the reference C# SURPASS-alpha Rouse validation at phi=0.035.

Matches the reference deliverable exactly:
  - phi = 0.035 (constant for all N)
  - Chain counts: N25=500, N50=500, N100=500, N250=200, N500=110
  - Eq sweeps:    N25=2000, N50=5000, N100=5000, N250=10000, N500=25000
  - Prod sweeps:  N25=5000, N50=5000, N100=5000, N250=5000,  N500=10000
  - Box sizes:    computed from phi=0.035

Usage:
    python3 -m rouse_model_python.run_reference_campaign \
        --device {cpu|gpu|mixed} --is_parallel {true|false}
"""

import sys
import os
import time
import random
import shutil
import pickle
import json
import datetime
import numpy as np
import torch
import multiprocessing as mp

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouse_model_python.config import (
    SimulationConfig, SEED, SIGMA, L0, format_phi,
)
from rouse_model_python.execution_policy import (
    parse_execution_args, ExecutionPolicy, SystemCapabilities,
)
from rouse_model_python.fast_simulation import FastRouseSimulation
from rouse_model_python.io_utils import (
    write_all_tsvs, generate_all_plots, write_validation_summary,
    write_readme, write_gitattributes, ensure_dir,
)


# ============================================================================
# Reference parameters from C# validation
# ============================================================================

PHI = 0.035
CHAIN_LENGTHS = [25, 50, 100, 250, 500]

# Exact parameters from the reference deliverable
REFERENCE_PARAMS = {
    25:  {'n_chains': 50,  'eq_sweeps': 500, 'prod_sweeps': 500, 'box': 269.6},
    50:  {'n_chains': 50,  'eq_sweeps': 500, 'prod_sweeps': 500, 'box': 339.7},
    100: {'n_chains': 50,  'eq_sweeps': 500, 'prod_sweeps': 500, 'box': 428.0},
    250: {'n_chains': 20,  'eq_sweeps': 500, 'prod_sweeps': 500, 'box': 428.0},
    500: {'n_chains': 10,  'eq_sweeps': 500, 'prod_sweeps': 500, 'box': 441.8},
}

# Output directory (separate from the phi-progression deliverable)
DELIVERABLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rouse_python_validation_reference_phi0035"
)


def verify_box_sizes():
    """Verify box sizes match phi=0.035 via L_box = (n_chains * N * sigma^3 / phi)^(1/3)."""
    print("\n=== Box Size Verification (phi=0.035) ===")
    sigma3 = SIGMA ** 3
    for N in CHAIN_LENGTHS:
        p = REFERENCE_PARAMS[N]
        box_calc = (p['n_chains'] * N * sigma3 / PHI) ** (1.0 / 3.0)
        phi_back = p['n_chains'] * N * sigma3 / (p['box'] ** 3)
        print(f"  N={N:>3}: chains={p['n_chains']:>3}, "
              f"box_ref={p['box']:.1f}, box_calc={box_calc:.1f}, "
              f"phi_back={phi_back:.4f} (target {PHI})")


def create_directory_tree(base_dir):
    """Create output directory tree."""
    dirs = [
        "01_static_properties",
        "02_dynamic_properties",
        "04_equilibration_evidence",
        "06_python_scripts",
    ]
    phi_str = format_phi(PHI)
    dirs.append(os.path.join("04_equilibration_evidence", f"phi_{phi_str}"))
    for N in CHAIN_LENGTHS:
        dirs.append(os.path.join("03_per_state_point", f"phi_{phi_str}", f"N{N}"))
        dirs.append(os.path.join("05_data", f"phi_{phi_str}", f"N{N}"))
    for d in dirs:
        ensure_dir(os.path.join(base_dir, d))
    print(f"  Created directory tree under {base_dir}")


def make_config(N: int, device: str) -> SimulationConfig:
    """Create SimulationConfig matching reference parameters exactly."""
    p = REFERENCE_PARAMS[N]
    cfg = SimulationConfig(
        N=N,
        n_chains=p['n_chains'],
        eq_sweeps=p['eq_sweeps'],
        prod_sweeps=p['prod_sweeps'],
        box_size=p['box'],
        seed=SEED,
        device=device,
        target_phi=PHI,
    )
    # Adaptive sample interval
    cfg.sample_interval = max(1, min(20, N // 10))
    return cfg


def run_single_N(N, seed_base, results_dir, use_fast_mode=True,
                  gpu_id=-1, use_batched_mode=False):
    """Run simulation for one chain length.

    Args:
        gpu_id: physical GPU index to use (-1 for CPU). Each worker sets
                CUDA_VISIBLE_DEVICES to isolate its GPU, then uses 'cuda:0'
                as the local device to avoid cross-process CUDA conflicts.
    """
    # Isolate GPU: make only this worker's GPU visible
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Must re-init CUDA after setting env var in spawned process
        torch_device = "cuda:0"  # local index after isolation
    else:
        torch_device = "cpu"

    random.seed(seed_base + N)
    np.random.seed(seed_base + N)
    torch.manual_seed(seed_base + N)
    if gpu_id >= 0:
        torch.cuda.manual_seed(seed_base + N)

    device_for_cfg = "cpu" if use_fast_mode else torch_device
    cfg = make_config(N, device=device_for_cfg)
    cfg.seed = seed_base + N
    cfg.use_batched_mode = use_batched_mode and not use_fast_mode

    print(f"\n[W-N{N}] Starting: chains={cfg.n_chains}, "
          f"eq={cfg.eq_sweeps}, prod={cfg.prod_sweeps}, "
          f"box={cfg.box_size:.1f}A, device={device_for_cfg}, "
          f"physical_gpu={gpu_id}", flush=True)

    if use_fast_mode:
        sim = FastRouseSimulation(cfg)
    else:
        from rouse_model_python.simulation import RouseSimulation
        sim = RouseSimulation(cfg)
    results = sim.run()

    # Write TSVs
    write_all_tsvs(results, DELIVERABLE_DIR, N, PHI)

    # Serialize for pickle
    serializable = {
        'N': results['N'],
        'n_chains': results['n_chains'],
        'final_R2': results['final_R2'].cpu().numpy(),
        'final_Rg2': results['final_Rg2'].cpu().numpy(),
        'g1': results['g1'],
        'gcm': results['gcm'],
        'gr': results['gr'],
        'sweep_data': results['sweep_data'],
    }

    pkl_path = os.path.join(results_dir, f"results_N{N}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(serializable, f)

    mean_R2 = float(results['final_R2'].mean())
    mean_Rg2 = float(results['final_Rg2'].mean())
    ratio = mean_R2 / mean_Rg2 if mean_Rg2 > 0 else 0
    print(f"[W-N{N}] Done: <R2>={mean_R2:.2f}, <Rg2>={mean_Rg2:.2f}, "
          f"R2/Rg2={ratio:.4f}", flush=True)
    return N


def run_all(policy, caps):
    """Run all 5 chain lengths and generate outputs."""
    print("=" * 70)
    print("ROUSE VALIDATION — REFERENCE RECREATION (phi=0.035)")
    print("Matching C# SURPASS-alpha deliverable parameters")
    print("=" * 70, flush=True)

    total_start = time.time()

    # Setup
    create_directory_tree(DELIVERABLE_DIR)
    verify_box_sizes()

    # Print config table
    print(f"\n{'N':>5} | {'chains':>6} | {'eq_sw':>7} | {'prod_sw':>8} | {'box(A)':>8} | {'beads':>7}")
    print("-" * 60)
    for N in CHAIN_LENGTHS:
        p = REFERENCE_PARAMS[N]
        print(f"{N:>5} | {p['n_chains']:>6} | {p['eq_sweeps']:>7} | "
              f"{p['prod_sweeps']:>8} | {p['box']:>8.1f} | {p['n_chains']*N:>7}")

    # Temp results dir
    results_dir = os.path.join(DELIVERABLE_DIR, "_tmp_results")
    ensure_dir(results_dir)

    # Determine simulation mode from policy
    use_fast = policy.torch_device == "cpu" and not policy.use_batched_mode
    use_batched = policy.use_batched_mode

    # Assign physical GPU IDs — one worker per GPU, heaviest gets dedicated.
    n_gpus = caps.n_gpus
    gpu_assignments = {}
    if policy.torch_device.startswith("cuda") and n_gpus > 0:
        # Give each chain length its own GPU (up to n_gpus).
        # If more chains than GPUs, lightest chains share.
        gpu_load = [0] * n_gpus
        sorted_by_beads = sorted(CHAIN_LENGTHS,
                                 key=lambda n: REFERENCE_PARAMS[n]['n_chains'] * n,
                                 reverse=True)
        for N in sorted_by_beads:
            lightest_gpu = min(range(n_gpus), key=lambda g: gpu_load[g])
            gpu_assignments[N] = lightest_gpu
            gpu_load[lightest_gpu] += REFERENCE_PARAMS[N]['n_chains'] * N
    else:
        for N in CHAIN_LENGTHS:
            gpu_assignments[N] = -1

    mode_str = "fast (numba+numpy)" if use_fast else f"PyTorch (multi-GPU: {n_gpus})"
    print(f"\nSimulation mode: {mode_str}")
    for N in CHAIN_LENGTHS:
        gpu = gpu_assignments[N]
        label = f"GPU {gpu}" if gpu >= 0 else "CPU"
        beads = REFERENCE_PARAMS[N]['n_chains'] * N
        print(f"  N={N:>3} ({beads:>5} beads) -> {label}")

    # Launch workers as subprocesses with CUDA_VISIBLE_DEVICES set per-process.
    # This ensures complete GPU isolation — each subprocess only sees its GPU.
    import subprocess
    print(f"\nLaunching {len(CHAIN_LENGTHS)} worker subprocesses...", flush=True)

    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "_run_single_worker.py")

    # Write a helper script that runs a single N
    with open(worker_script, 'w') as f:
        f.write('''"""Worker subprocess: runs one chain length on one GPU."""
import sys, os, pickle, random, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouse_model_python.config import SimulationConfig, SIGMA, L0, SEED
from rouse_model_python.simulation import RouseSimulation
from rouse_model_python.io_utils import write_all_tsvs, ensure_dir

N = int(sys.argv[1])
results_dir = sys.argv[2]
deliverable_dir = sys.argv[3]
params = json.loads(sys.argv[4])

seed_base = SEED
random.seed(seed_base + N)
np.random.seed(seed_base + N)
torch.manual_seed(seed_base + N)
if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed(seed_base + N)

device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

cfg = SimulationConfig(
    N=N,
    n_chains=params["n_chains"],
    eq_sweeps=params["eq_sweeps"],
    prod_sweeps=params["prod_sweeps"],
    box_size=params["box"],
    seed=seed_base + N,
    device=device,
    target_phi=0.035,
    sample_interval=max(1, min(20, N // 10)),
    use_batched_mode=params.get("use_batched", torch.cuda.device_count() > 0),
)

print(f"[W-N{N}] device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}", flush=True)

sim = RouseSimulation(cfg)
results = sim.run()

write_all_tsvs(results, deliverable_dir, N, 0.035)

serializable = {
    "N": results["N"],
    "n_chains": results["n_chains"],
    "final_R2": results["final_R2"].cpu().numpy(),
    "final_Rg2": results["final_Rg2"].cpu().numpy(),
    "g1": results["g1"],
    "gcm": results["gcm"],
    "gr": results["gr"],
    "sweep_data": results["sweep_data"],
}
pkl_path = os.path.join(results_dir, f"results_N{N}.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(serializable, f)

mean_R2 = float(results["final_R2"].mean())
mean_Rg2 = float(results["final_Rg2"].mean())
ratio = mean_R2 / mean_Rg2 if mean_Rg2 > 0 else 0
print(f"[W-N{N}] Done: <R2>={mean_R2:.2f}, <Rg2>={mean_Rg2:.2f}, R2/Rg2={ratio:.4f}", flush=True)
''')

    # Launch all workers as isolated subprocesses.
    # Stdout goes to log files (not pipes) to avoid pipe-buffer deadlocks.
    log_dir = os.path.join(DELIVERABLE_DIR, "_worker_logs")
    ensure_dir(log_dir)

    procs = []
    log_files = []
    for N in CHAIN_LENGTHS:
        gpu_id = gpu_assignments[N]
        env = os.environ.copy()
        if gpu_id >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        worker_params = dict(REFERENCE_PARAMS[N], use_batched=use_batched)
        params_json = json.dumps(worker_params)
        cmd = [sys.executable, worker_script, str(N), results_dir,
               DELIVERABLE_DIR, params_json]
        log_path = os.path.join(log_dir, f"worker_N{N}.log")
        log_f = open(log_path, 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f,
                             stderr=subprocess.STDOUT)
        procs.append((N, gpu_id, p))
        log_files.append(log_f)
        print(f"  Spawned N={N} (PID {p.pid}) on GPU {gpu_id} -> {log_path}",
              flush=True)

    # Poll until all workers complete (no pipe blocking)
    import time as _time
    completed = set()
    while len(completed) < len(procs):
        _time.sleep(30)
        for N, gpu_id, p in procs:
            if N in completed:
                continue
            ret = p.poll()
            if ret is not None:
                completed.add(N)
                status = "finished successfully" if ret == 0 else f"FAILED (exit {ret})"
                print(f">>> Worker N={N} {status}.", flush=True)
        pending = [N for N, _, _ in procs if N not in completed]
        if pending:
            print(f"    Still running: {pending}", flush=True)

    # Close log files
    for f in log_files:
        f.close()

    # Print worker logs
    for N, _, p in procs:
        log_path = os.path.join(log_dir, f"worker_N{N}.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            # Print last 5 lines of each log
            print(f"\n--- Worker N={N} (last 5 lines) ---")
            for line in lines[-5:]:
                print(f"  {line.rstrip()}")
    print(flush=True)

    # Assemble results
    print("\n" + "=" * 70)
    print("ASSEMBLING RESULTS")
    print("=" * 70)

    phi_str = format_phi(PHI)
    all_results = {PHI: {}}

    for N in CHAIN_LENGTHS:
        pkl_path = os.path.join(results_dir, f"results_N{N}.pkl")
        if not os.path.exists(pkl_path):
            print(f"WARNING: Missing results for N={N}")
            continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        data['final_R2'] = torch.from_numpy(data['final_R2'])
        data['final_Rg2'] = torch.from_numpy(data['final_Rg2'])
        all_results[PHI][N] = data

    shutil.rmtree(results_dir, ignore_errors=True)

    # Generate plots and analysis
    print("\n" + "=" * 70)
    print("GENERATING PLOTS AND ANALYSIS")
    print("=" * 70)

    generate_all_plots(all_results, DELIVERABLE_DIR)

    # Validation summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    summary = write_validation_summary(all_results, DELIVERABLE_DIR)

    # Print results table matching reference format
    print("\n=== Static Properties ===")
    print(f"{'N':>5} | {'<R2> (A^2)':>12} | {'<Rg2> (A^2)':>12} | {'R2/Rg2':>8}")
    print("-" * 50)
    if PHI in all_results:
        for N in CHAIN_LENGTHS:
            if N in all_results[PHI]:
                d = all_results[PHI][N]
                r2 = float(d['final_R2'].mean())
                rg2 = float(d['final_Rg2'].mean())
                ratio = r2 / rg2 if rg2 > 0 else 0
                print(f"{N:>5} | {r2:>12.2f} | {rg2:>12.2f} | {ratio:>8.4f}")

    # Print phi=0.035 dilute-limit metrics
    if PHI in summary.get("per_phi", {}):
        metrics = summary["per_phi"][PHI]
        print("\n=== Scaling Exponents ===")
        for key in sorted(metrics.keys()):
            if key.startswith('_'):
                continue
            entry = metrics[key]
            if isinstance(entry, dict) and "pass" in entry:
                status = "PASS" if entry["pass"] else "FAIL"
                r2_fit = entry.get('r2_fit', 'N/A')
                print(f"  {key}: measured={entry.get('measured', 'N/A')}, "
                      f"theory={entry.get('theory', 'N/A')}, "
                      f"R2_fit={r2_fit} -> {status}")

    # Write report files
    write_readme(all_results, summary, DELIVERABLE_DIR)
    write_gitattributes(DELIVERABLE_DIR)

    # Copy scripts
    source_dir = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(DELIVERABLE_DIR, "06_python_scripts")
    ensure_dir(dest)
    py_files = [f for f in os.listdir(source_dir)
                if f.endswith('.py') and not f.startswith('__')]
    for f in py_files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(dest, f))
    print(f"  Copied {len(py_files)} scripts to 06_python_scripts/")

    # Write requirements and iteration log
    reqs_path = os.path.join(DELIVERABLE_DIR, "requirements_list.md")
    with open(reqs_path, 'w') as f:
        f.write("# Requirements List — Reference Recreation (phi=0.035)\n\n")
        f.write("Recreation of C# SURPASS-alpha validation at constant phi=0.035\n")
        f.write("using the Python Rouse model codebase.\n\n")
        f.write("## Parameters\n")
        f.write(f"- phi = {PHI}\n")
        f.write(f"- Chain lengths: {CHAIN_LENGTHS}\n")
        f.write(f"- seed = {SEED}\n\n")

    log_path = os.path.join(DELIVERABLE_DIR, "iteration_log.txt")
    with open(log_path, 'w') as f:
        f.write("Rouse Validation — Reference Recreation Log\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"phi = {PHI} (constant)\n\n")
        for N in CHAIN_LENGTHS:
            p = REFERENCE_PARAMS[N]
            f.write(f"[W-N{N}] chains={p['n_chains']}, "
                    f"eq={p['eq_sweeps']}, prod={p['prod_sweeps']}, "
                    f"box={p['box']}A\n")

    # Verify outputs
    total_time = time.time() - total_start
    print(f"\nTotal wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"All outputs written to: {DELIVERABLE_DIR}")

    # Count files
    n_tsv = 0
    n_png = 0
    for root, dirs, files in os.walk(DELIVERABLE_DIR):
        for f in files:
            if f.endswith('.tsv'):
                n_tsv += 1
            elif f.endswith('.png'):
                n_png += 1
    print(f"  {n_tsv} TSV files, {n_png} PNG plots generated")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    policy, caps, cli_args = parse_execution_args()
    run_all(policy, caps)
