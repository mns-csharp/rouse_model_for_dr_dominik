"""DeviceDispatcher — cost-aware GPU/CPU partitioning and round-robin assignment.

Implements the dispatch logic for --device=mixed (cost-aware partition) and
--device=gpu (round-robin across multiple GPUs).
"""

from typing import List, Optional

import torch

from rouse_model_python.src.libs.execution.execution_policy import ExecutionPolicy
from rouse_model_python.src.libs.execution.sim_work_item import SimWorkItem
from rouse_model_python.src.libs.execution.system_capabilities import SystemCapabilities
from rouse_model_python.src.libs.execution.vram_estimator import VRAMEstimator


class DeviceDispatcher:
    GPU_SPEEDUP_FACTOR = 8.0
    GPU_OVERHEAD_BEADS = 500

    @staticmethod
    def estimate_sim_cost(N: int, n_chains: int, sweeps: int) -> float:
        return float(N) * n_chains

    @classmethod
    def partition_mixed_workload(
            cls,
            work_items: List[SimWorkItem],
            caps: SystemCapabilities,
            threshold_k: Optional[float] = None,
            gpu_speedup: Optional[float] = None,
            gpu_overhead_beads: Optional[int] = None,
    ) -> List[SimWorkItem]:
        gpu_speedup = gpu_speedup if gpu_speedup is not None else cls.GPU_SPEEDUP_FACTOR
        gpu_overhead_beads = (gpu_overhead_beads if gpu_overhead_beads is not None
                              else cls.GPU_OVERHEAD_BEADS)

        if caps.n_gpus == 0:
            for w in work_items:
                w.device = "cpu"
            return work_items

        for w in work_items:
            w.cost = cls.estimate_sim_cost(w.N, w.n_chains, w.sweeps)

        gpu_free_mb: List[float] = []
        for i in range(caps.n_gpus):
            try:
                free, _ = torch.cuda.mem_get_info(i)
                gpu_free_mb.append(free / (1024 ** 2))
            except Exception:
                gpu_free_mb.append(0.0)
        min_free_mb = min(gpu_free_mb) if gpu_free_mb else 0.0

        if threshold_k is not None:
            gpu_assigned: List[SimWorkItem] = []
            cpu_assigned: List[SimWorkItem] = []
            for w in work_items:
                vram_needed = VRAMEstimator.estimate_vram_mb(w.N, w.n_chains)
                if w.cost > threshold_k and vram_needed <= min_free_mb:
                    gpu_assigned.append(w)
                else:
                    cpu_assigned.append(w)
        else:
            gpu_eligible: List[SimWorkItem] = []
            cpu_forced: List[SimWorkItem] = []
            for w in work_items:
                total_beads = w.N * w.n_chains
                vram_needed = VRAMEstimator.estimate_vram_mb(w.N, w.n_chains)
                if total_beads < gpu_overhead_beads:
                    cpu_forced.append(w)
                elif vram_needed > min_free_mb:
                    n_sub = VRAMEstimator.compute_gpu_sub_batches(w.N, w.n_chains)
                    w.sub_batches = n_sub
                    gpu_eligible.append(w)
                else:
                    gpu_eligible.append(w)

            if not gpu_eligible:
                for w in work_items:
                    w.device = "cpu"
                return work_items

            gpu_eligible.sort(key=lambda w: w.cost, reverse=True)
            gpu_time = 0.0
            cpu_time = sum(w.cost for w in cpu_forced)
            gpu_assigned = []
            cpu_assigned = list(cpu_forced)

            for w in gpu_eligible:
                gpu_wall = w.cost / gpu_speedup
                if gpu_time + gpu_wall <= cpu_time + w.cost:
                    gpu_assigned.append(w)
                    gpu_time += gpu_wall
                else:
                    cpu_assigned.append(w)
                    cpu_time += w.cost

        for w in cpu_assigned:
            w.device = "cpu"
        for i, w in enumerate(gpu_assigned):
            w.device = f"cuda:{i % caps.n_gpus}"

        return work_items

    @staticmethod
    def assign_devices_for_simulations(n_simulations: int,
                                       caps: SystemCapabilities,
                                       policy: ExecutionPolicy) -> List[str]:
        if policy.device == "gpu" and caps.n_gpus > 1:
            return [f"cuda:{i % caps.n_gpus}" for i in range(n_simulations)]
        elif policy.device == "mixed" and caps.n_gpus > 0:
            return [f"cuda:{i % max(1, caps.n_gpus)}" for i in range(n_simulations)]
        else:
            return [policy.torch_device] * n_simulations
