"""
Execution policy: mandatory CLI arguments, system capability query, and
device/parallel dispatch logic.

Every entry point MUST call parse_execution_args() to obtain device and
is_parallel settings. There are NO defaults -- both --device and
--is_parallel are required command-line arguments.

System capability (GPU count, CPU cores) is queried at startup for
diagnostics and dispatch but does NOT override the user's --device choice.
"""

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


def cuda_available() -> bool:
    """Check CUDA availability robustly.

    ``torch.cuda.is_available()`` can return False in container environments
    where the CUDA management interface works but the runtime init path fails.
    ``torch.cuda.device_count()`` often succeeds in those same environments,
    so we use it as the primary check.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return torch.cuda.device_count() > 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# System capability query
# ---------------------------------------------------------------------------

@dataclass
class SystemCapabilities:
    """Hardware capabilities detected at startup."""
    n_gpus: int
    gpu_names: List[str]
    n_cpu_cores: int

    def report(self) -> str:
        lines = [
            f"  CPU cores available: {self.n_cpu_cores}",
            f"  GPU devices attached: {self.n_gpus}",
        ]
        for i, name in enumerate(self.gpu_names):
            try:
                vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                lines.append(f"    GPU {i}: {name} ({vram:.1f} GB)")
            except Exception:
                lines.append(f"    GPU {i}: {name}")
        return "\n".join(lines)


def _gpu_count_from_nvidia_smi() -> Tuple[int, List[str]]:
    """Detect GPUs via nvidia-smi when torch.cuda is unavailable."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            names = [line.strip() for line in result.stdout.strip().splitlines()]
            return len(names), names
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0, []


def query_system_capabilities() -> SystemCapabilities:
    """Query OS environment for GPU count and CPU core count."""
    n_cpu = os.cpu_count() or 1

    # torch.cuda.device_count() can succeed even when is_available() returns
    # False (e.g. inside containers with restricted /dev access that still
    # expose the NVIDIA management interface).  Try several detection paths.
    n_gpus = 0
    gpu_names: List[str] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            n_gpus = torch.cuda.device_count()
        except Exception:
            n_gpus = 0

    if n_gpus > 0:
        try:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        except Exception:
            # torch can count devices but not query names; fall back to smi
            _, gpu_names = _gpu_count_from_nvidia_smi()
    else:
        # Last resort: ask nvidia-smi directly
        n_gpus, gpu_names = _gpu_count_from_nvidia_smi()

    return SystemCapabilities(n_gpus=n_gpus, gpu_names=gpu_names, n_cpu_cores=n_cpu)


# ---------------------------------------------------------------------------
# Execution policy
# ---------------------------------------------------------------------------

@dataclass
class ExecutionPolicy:
    """Resolved execution settings from --device and --is_parallel."""
    device: str           # "cpu", "gpu", or "mixed" (user's original choice)
    is_parallel: bool
    torch_device: str     # resolved torch device string ("cpu" or "cuda")
    step_size: int        # MC move batch granularity (1 for serial, >1 for parallel)
    use_batched_mode: bool  # whether to use batched GPU proposals
    warning: Optional[str] = None

    def report(self) -> str:
        lines = [
            f"  Device: {self.device}",
            f"  Parallel: {self.is_parallel}",
            f"  Torch device: {self.torch_device}",
            f"  Step size: {self.step_size}",
            f"  Batched mode: {self.use_batched_mode}",
        ]
        if self.warning:
            lines.append(f"  WARNING: {self.warning}")
        return "\n".join(lines)


def resolve_execution_policy(device: str, is_parallel: bool,
                             caps: SystemCapabilities,
                             force: bool = False) -> ExecutionPolicy:
    """Apply the execution policy table.

    --device | --is_parallel | Behaviour
    ---------|---------------|------------------------------------------
    cpu      | false         | Serial CPU, step_size=1
    cpu      | true          | Parallel CPU, step_size>1
    gpu      | false         | WARN, serial GPU (suppress with --force)
    gpu      | true          | Parallel GPU, step_size>1, batched mode
    mixed    | false         | WARN, falls back to CPU-only serial
    mixed    | true          | Hybrid CPU+GPU dispatch
    """
    warning = None

    if device == "cpu":
        torch_device = "cpu"
        step_size = 20 if is_parallel else 1
        use_batched = False

    elif device == "gpu":
        if caps.n_gpus == 0:
            print("ERROR: --device gpu requested but no CUDA GPU is available.",
                  file=sys.stderr)
            sys.exit(1)
        torch_device = "cuda"
        if not is_parallel:
            warning = ("gpu + is_parallel=false: serial GPU execution. "
                       "Use --force to suppress this warning.")
            if not force:
                print(f"WARNING: {warning}", file=sys.stderr)
            step_size = 1
            use_batched = True
        else:
            step_size = 20
            use_batched = True

    elif device == "mixed":
        if not is_parallel:
            warning = ("mixed + is_parallel=false: falling back to CPU-only serial.")
            print(f"WARNING: {warning}", file=sys.stderr)
            torch_device = "cpu"
            step_size = 1
            use_batched = False
        else:
            # Hybrid: GPU for compute-heavy chains, CPU for lightweight
            torch_device = "cuda" if caps.n_gpus > 0 else "cpu"
            step_size = 20
            use_batched = caps.n_gpus > 0

    else:
        print(f"ERROR: Invalid --device value: {device!r}. "
              f"Must be one of: cpu, gpu, mixed.", file=sys.stderr)
        sys.exit(1)

    return ExecutionPolicy(
        device=device,
        is_parallel=is_parallel,
        torch_device=torch_device,
        step_size=step_size,
        use_batched_mode=use_batched,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# CLI argument parsing (mandatory --device and --is_parallel)
# ---------------------------------------------------------------------------

def add_execution_args(parser: argparse.ArgumentParser) -> None:
    """Add the mandatory --device and --is_parallel arguments to a parser."""
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cpu", "gpu", "mixed"],
        help="Device target for execution: cpu, gpu, or mixed. REQUIRED.",
    )
    parser.add_argument(
        "--is_parallel",
        type=str,
        required=True,
        choices=["true", "false"],
        help="Whether to use parallel execution strategy: true or false. REQUIRED.",
    )
    parser.add_argument(
        "--batched",
        type=str,
        required=False,
        choices=["true", "false"],
        default=None,
        help="Override batched mode: true or false. If omitted, auto-resolved from device/parallel.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Suppress warnings for non-optimal device/parallel combinations.",
    )


def parse_execution_args(argv: Optional[List[str]] = None,
                         extra_args_fn=None) -> Tuple[ExecutionPolicy, SystemCapabilities, argparse.Namespace]:
    """Parse mandatory execution arguments, query system, resolve policy.

    Args:
        argv: command-line arguments (defaults to sys.argv[1:])
        extra_args_fn: optional callable(parser) to add extra arguments

    Returns:
        (policy, capabilities, namespace) tuple
    """
    parser = argparse.ArgumentParser(
        description="Rouse model MC simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_execution_args(parser)
    if extra_args_fn is not None:
        extra_args_fn(parser)

    args = parser.parse_args(argv)

    # System capability query
    caps = query_system_capabilities()
    print("System capabilities:")
    print(caps.report())

    # Resolve execution policy
    is_parallel = args.is_parallel.lower() == "true"
    policy = resolve_execution_policy(args.device, is_parallel, caps,
                                      force=args.force)

    # Apply --batched override if provided
    if args.batched is not None:
        policy.use_batched_mode = args.batched.lower() == "true"

    print("\nExecution policy:")
    print(policy.report())

    return policy, caps, args


# ---------------------------------------------------------------------------
# Multi-GPU simulation distribution
# ---------------------------------------------------------------------------

@dataclass
class SimWorkItem:
    """Describes a simulation for cost-aware device assignment."""
    index: int          # position in the work list (for stable ordering)
    N: int              # beads per chain
    n_chains: int       # number of chains
    sweeps: int         # total sweeps (eq + prod)
    cost: float = 0.0   # estimated compute cost (filled by estimate_sim_cost)
    device: str = ""    # assigned device (filled by partition_mixed_workload)


# Empirical GPU-vs-CPU speed ratio.  GPU kernel launch overhead means tiny
# workloads run *slower* on GPU than CPU; large workloads run much faster.
# These ratios were measured on A100 / Xeon Gold and are intentionally
# conservative so that borderline cases stay on CPU.
_GPU_SPEEDUP_FACTOR = 8.0   # GPU is ~8x faster for large workloads
_GPU_OVERHEAD_BEADS = 500    # below this total-bead count, GPU overhead dominates


def estimate_sim_cost(N: int, n_chains: int, sweeps: int) -> float:
    """Estimate relative compute cost for a simulation.

    Cost scales as N^2 * n_chains * sweeps.  The N^2 term reflects pairwise
    bead interactions within the energy kernel (cell-list bounded, but still
    quadratic in segment size ≈ N for short chains and grows with N).
    """
    return float(N * N) * n_chains * sweeps


def partition_mixed_workload(
    work_items: List[SimWorkItem],
    caps: SystemCapabilities,
    gpu_speedup: float = _GPU_SPEEDUP_FACTOR,
    gpu_overhead_beads: int = _GPU_OVERHEAD_BEADS,
) -> List[SimWorkItem]:
    """Cost-aware GPU/CPU partitioning for --device=mixed.

    Strategy:
    1. Compute a cost estimate for each simulation (N^2 * n_chains * sweeps).
    2. Exclude tiny workloads (total beads < gpu_overhead_beads) — they always
       go to CPU because GPU kernel launch overhead exceeds any speedup.
    3. Check VRAM: simulations whose estimated VRAM exceeds available headroom
       are forced to CPU.
    4. Greedily assign the remaining simulations to GPU or CPU to equalize
       total *effective* work on each side.  GPU work is divided by
       gpu_speedup to reflect wall-clock time rather than raw cost.

    Returns the same list with .device populated on every item.
    """
    if caps.n_gpus == 0:
        for w in work_items:
            w.device = "cpu"
        return work_items

    # Step 1: fill cost estimates
    for w in work_items:
        w.cost = estimate_sim_cost(w.N, w.n_chains, w.sweeps)

    # Step 2: classify items that *must* go to CPU
    gpu_eligible: List[SimWorkItem] = []
    cpu_forced: List[SimWorkItem] = []

    # Query per-GPU free VRAM once
    gpu_free_mb: List[float] = []
    for i in range(caps.n_gpus):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            gpu_free_mb.append(free / (1024 ** 2))
        except Exception:
            gpu_free_mb.append(0.0)
    # Use the minimum across GPUs as the per-sim budget (conservative)
    min_free_mb = min(gpu_free_mb) if gpu_free_mb else 0.0

    for w in work_items:
        total_beads = w.N * w.n_chains
        vram_needed = estimate_vram_mb(w.N, w.n_chains)
        if total_beads < gpu_overhead_beads:
            cpu_forced.append(w)
        elif vram_needed > min_free_mb:
            cpu_forced.append(w)
        else:
            gpu_eligible.append(w)

    # If nothing is GPU-eligible, everything goes to CPU
    if not gpu_eligible:
        for w in work_items:
            w.device = "cpu"
        return work_items

    # Step 3: greedy balanced partition
    # Sort eligible items by cost descending (largest-first-fit)
    gpu_eligible.sort(key=lambda w: w.cost, reverse=True)

    gpu_time = 0.0   # effective wall-clock units on GPU
    cpu_time = sum(w.cost for w in cpu_forced)  # CPU already has forced items

    gpu_assigned: List[SimWorkItem] = []
    cpu_assigned: List[SimWorkItem] = list(cpu_forced)

    for w in gpu_eligible:
        gpu_wall = w.cost / gpu_speedup
        # Assign to whichever side has less accumulated wall-clock time
        if gpu_time + gpu_wall <= cpu_time + w.cost:
            gpu_assigned.append(w)
            gpu_time += gpu_wall
        else:
            cpu_assigned.append(w)
            cpu_time += w.cost

    # Step 4: assign device strings
    for w in cpu_assigned:
        w.device = "cpu"

    # Round-robin GPU assignments across available GPUs
    for i, w in enumerate(gpu_assigned):
        w.device = f"cuda:{i % caps.n_gpus}"

    return work_items


def assign_devices_for_simulations(n_simulations: int,
                                   caps: SystemCapabilities,
                                   policy: ExecutionPolicy) -> List[str]:
    """Assign a torch device string to each simulation for multi-GPU dispatch.

    Policy:
    - If device=gpu and multiple GPUs: round-robin across GPUs
    - If device=mixed: use partition_mixed_workload() for cost-aware dispatch
    - If device=cpu or single GPU: all get the same device

    Note: for mixed mode with full cost-aware partitioning, prefer calling
    partition_mixed_workload() directly with SimWorkItem details.  This
    function provides a simpler interface when per-sim details aren't available.
    """
    if policy.device == "gpu" and caps.n_gpus > 1:
        # Round-robin across available GPUs
        return [f"cuda:{i % caps.n_gpus}" for i in range(n_simulations)]
    elif policy.device == "mixed" and caps.n_gpus > 0:
        return [f"cuda:{i % max(1, caps.n_gpus)}" for i in range(n_simulations)]
    else:
        return [policy.torch_device] * n_simulations


def check_vram_headroom(device_idx: int, required_mb: float) -> bool:
    """Check if a GPU has enough free VRAM for a simulation batch.

    Returns True if enough headroom exists or if running on CPU.
    """
    if torch.cuda.device_count() == 0:
        return True
    if device_idx >= torch.cuda.device_count():
        return False
    free, total = torch.cuda.mem_get_info(device_idx)
    free_mb = free / (1024 ** 2)
    return free_mb >= required_mb


def estimate_vram_mb(N: int, n_chains: int) -> float:
    """Rough estimate of VRAM needed for a simulation (MB).

    Accounts for position tensors, energy matrices, and overhead.
    """
    # Position tensor: n_chains * N * 3 * 4 bytes (float32)
    pos_bytes = n_chains * N * 3 * 4
    # Energy matrix overhead: ~N * N * 4 bytes per batch
    energy_bytes = N * N * 4
    # Cell list, proposals, etc: ~2x position
    overhead = pos_bytes * 2
    total = pos_bytes + energy_bytes + overhead
    return total / (1024 ** 2) * 2  # 2x safety factor
