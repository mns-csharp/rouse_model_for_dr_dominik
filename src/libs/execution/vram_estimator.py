"""VRAMEstimator — GPU memory estimates and OOM-prevention helpers."""

import math

import torch


class VRAMEstimator:
    @staticmethod
    def estimate_vram_mb(N: int, n_chains: int) -> float:
        pos_bytes = n_chains * N * 3 * 4
        energy_bytes = N * N * 4
        overhead = pos_bytes * 2
        total = pos_bytes + energy_bytes + overhead
        return total / (1024 ** 2) * 2

    @staticmethod
    def check_vram_headroom(device_idx: int, required_mb: float) -> bool:
        if torch.cuda.device_count() == 0:
            return True
        if device_idx >= torch.cuda.device_count():
            return False
        free, _ = torch.cuda.mem_get_info(device_idx)
        free_mb = free / (1024 ** 2)
        return free_mb >= required_mb

    @classmethod
    def compute_gpu_sub_batches(cls, N: int, n_chains: int, device_idx: int = 0) -> int:
        if torch.cuda.device_count() == 0:
            return 1
        if device_idx >= torch.cuda.device_count():
            return 1
        try:
            free, _ = torch.cuda.mem_get_info(device_idx)
            free_mb = free / (1024 ** 2)
        except Exception:
            return 1

        full_vram = cls.estimate_vram_mb(N, n_chains)
        if full_vram <= free_mb:
            return 1

        for k in range(2, n_chains + 1):
            sub_chains = math.ceil(n_chains / k)
            if cls.estimate_vram_mb(N, sub_chains) <= free_mb:
                return k
        return n_chains
