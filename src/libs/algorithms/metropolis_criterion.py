"""MetropolisCriterion — standard Metropolis acceptance test with overflow guards."""

import math
import torch


class MetropolisCriterion:
    @staticmethod
    def accept(delta_e: float, kBT: float, gen: torch.Generator,
               device: torch.device, dtype: torch.dtype,
               rand_pool=None) -> bool:
        if delta_e <= 0.0:
            return True
        exponent = -delta_e / kBT
        if exponent <= -745.0:
            return False
        if exponent >= 709.0:
            return True
        prob = math.exp(exponent)
        if rand_pool is not None:
            r = rand_pool.next()
        else:
            r = torch.rand(1, generator=gen, dtype=dtype, device=device).item()
        return r < prob
