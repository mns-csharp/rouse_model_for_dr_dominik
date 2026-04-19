"""RandPool — pre-generated random number pool to avoid per-call GPU→CPU sync."""

import torch


class RandPool:
    def __init__(self, gen: torch.Generator, device: torch.device,
                 dtype: torch.dtype, initial_size: int = 10000):
        self._gen = gen
        self._device = device
        self._dtype = dtype
        self._pool = torch.rand(
            initial_size, generator=gen, dtype=dtype, device=device).tolist()
        self._idx = 0

    def next(self) -> float:
        if self._idx >= len(self._pool):
            more = torch.rand(
                5000, generator=self._gen, dtype=self._dtype,
                device=self._device).tolist()
            self._pool.extend(more)
        val = self._pool[self._idx]
        self._idx += 1
        return val
