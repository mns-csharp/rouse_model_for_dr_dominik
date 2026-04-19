"""SpeedupComputer — Migacz speedup + GPU/CPU speedup ratios."""

import math
from typing import Dict, Tuple


class SpeedupComputer:
    ALL_N = ("25", "50", "100")

    @staticmethod
    def _t(key, groups):
        row = groups.get(key)
        if row is None:
            return float("nan")
        try:
            return float(row["total_wall_s"])
        except (ValueError, KeyError):
            return float("nan")

    @classmethod
    def compute(cls, groups) -> dict:
        s_migacz: Dict[Tuple[str, str], float] = {}
        s_gpu: Dict[Tuple[str, str], float] = {}
        for N in cls.ALL_N:
            for dev in ("cpu", "gpu"):
                tc = cls._t((N, "conventional", dev), groups)
                tm = cls._t((N, "multistep", dev), groups)
                s_migacz[(N, dev)] = (tc / tm if (math.isfinite(tc) and math.isfinite(tm) and tm > 0)
                                      else float("nan"))
            for alg in ("multistep", "conventional"):
                tcpu = cls._t((N, alg, "cpu"), groups)
                tgpu = cls._t((N, alg, "gpu"), groups)
                s_gpu[(N, alg)] = (tcpu / tgpu if (math.isfinite(tcpu) and math.isfinite(tgpu) and tgpu > 0)
                                   else float("nan"))
        return {"migacz": s_migacz, "gpu": s_gpu}
