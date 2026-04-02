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
    print("\nExecution policy:")
    print(policy.report())

    return policy, caps, args


# ---------------------------------------------------------------------------
# Multi-GPU simulation distribution
# ---------------------------------------------------------------------------

def assign_devices_for_simulations(n_simulations: int,
                                   caps: SystemCapabilities,
                                   policy: ExecutionPolicy) -> List[str]:
    """Assign a torch device string to each simulation for multi-GPU dispatch.

    Policy:
    - If device=gpu and multiple GPUs: round-robin across GPUs
    - If device=mixed: large-N sims go to GPU, small-N to CPU
    - If device=cpu or single GPU: all get the same device
    """
    if policy.device == "gpu" and caps.n_gpus > 1:
        # Round-robin across available GPUs
        return [f"cuda:{i % caps.n_gpus}" for i in range(n_simulations)]
    elif policy.device == "mixed" and caps.n_gpus > 0:
        # GPU for all if parallel, dispatch logic can be refined per-N
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
