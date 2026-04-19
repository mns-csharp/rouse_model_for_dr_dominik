"""ExecutionPolicy — resolved device/parallel policy + CLI arg helper."""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

from rouse_model_python.src.libs.execution.system_capabilities import SystemCapabilities


@dataclass
class ExecutionPolicy:
    device: str
    is_parallel: bool
    torch_device: str
    step_size: int
    use_batched_mode: bool
    mixed_threshold: Optional[float] = None
    warning: Optional[str] = None

    def report(self) -> str:
        lines = [
            f"  Device: {self.device}",
            f"  Parallel: {self.is_parallel}",
            f"  Torch device: {self.torch_device}",
            f"  Step size: {self.step_size}",
            f"  Batched mode: {self.use_batched_mode}",
        ]
        if self.mixed_threshold is not None:
            lines.append(f"  Mixed threshold K: {self.mixed_threshold}")
        if self.warning:
            lines.append(f"  WARNING: {self.warning}")
        return "\n".join(lines)

    @classmethod
    def resolve(cls, device: str, is_parallel: bool,
                caps: SystemCapabilities, force: bool = False) -> "ExecutionPolicy":
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
                warning = ("gpu + is_parallel=false: single-trial sweep granularity "
                           "(step_size=1). Batched-mode multistep may still run per "
                           "app overrides. Use --force to suppress this warning.")
                if not force:
                    print(f"WARNING: {warning}", file=sys.stderr)
                step_size = 1
                use_batched = True
            else:
                step_size = 20
                use_batched = True
        elif device == "mixed":
            if caps.n_gpus == 0:
                print("ERROR: --device mixed requested but no CUDA GPU is available.",
                      file=sys.stderr)
                sys.exit(1)
            if not is_parallel:
                warning = "mixed + is_parallel=false: falling back to CPU-only serial."
                print(f"WARNING: {warning}", file=sys.stderr)
                torch_device = "cpu"
                step_size = 1
                use_batched = False
            else:
                torch_device = "cuda"
                step_size = 20
                use_batched = True
        else:
            print(f"ERROR: Invalid --device value: {device!r}.", file=sys.stderr)
            sys.exit(1)
        return cls(device=device, is_parallel=is_parallel,
                   torch_device=torch_device, step_size=step_size,
                   use_batched_mode=use_batched, warning=warning)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--device", type=str, required=True,
                            choices=["cpu", "gpu", "mixed"])
        parser.add_argument("--is_parallel", type=str, required=True,
                            choices=["true", "false"])
        parser.add_argument("--batched", type=str, required=False,
                            choices=["true", "false"], default=None)
        parser.add_argument("--force", action="store_true", default=False)
        parser.add_argument("--mixed-threshold", type=float, required=False,
                            default=None)
        parser.add_argument("--algorithm", type=str, required=False,
                            default="multistep",
                            choices=["multistep", "conventional"])
