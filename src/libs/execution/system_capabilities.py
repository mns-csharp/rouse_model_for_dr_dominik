"""SystemCapabilities — hardware discovery (GPU count, CPU cores, device names)."""

from dataclasses import dataclass, field
from typing import List
import os
import subprocess
import warnings

import torch


@dataclass
class SystemCapabilities:
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

    @staticmethod
    def _gpu_count_from_nvidia_smi():
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

    @classmethod
    def query(cls) -> "SystemCapabilities":
        n_cpu = os.cpu_count() or 1
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
                _, gpu_names = cls._gpu_count_from_nvidia_smi()
        else:
            n_gpus, gpu_names = cls._gpu_count_from_nvidia_smi()
        return cls(n_gpus=n_gpus, gpu_names=gpu_names, n_cpu_cores=n_cpu)

    @staticmethod
    def cuda_available() -> bool:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return torch.cuda.device_count() > 0
            except Exception:
                return False
