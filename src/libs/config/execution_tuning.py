"""ExecutionTuning — dataclass loaded from configs/execution.toml."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExecutionTuning:
    device_default: str
    step_size_serial: int
    step_size_parallel: int
    gpu_speedup_factor: float
    gpu_overhead_beads: int
    torch_compile_mode: str

    @classmethod
    def from_toml(cls, path: Path) -> "ExecutionTuning":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(
            device_default=data["device"]["default"],
            step_size_serial=data["parallel"]["step_size_serial"],
            step_size_parallel=data["parallel"]["step_size_parallel"],
            gpu_speedup_factor=data["gpu_tuning"]["speedup_factor"],
            gpu_overhead_beads=data["gpu_tuning"]["overhead_beads"],
            torch_compile_mode=data["torch_compile"]["mode"],
        )
