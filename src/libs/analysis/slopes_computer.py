"""SlopesComputer — log-log scaling fit for wall-time vs N."""

import math
from typing import Dict, Tuple


class SlopesComputer:
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
    def compute(cls, groups) -> Dict[Tuple[str, str], float]:
        slopes = {}
        for alg in ("multistep", "conventional"):
            for dev in ("cpu", "gpu"):
                log_n = []
                log_t = []
                for N in cls.ALL_N:
                    t = cls._t((N, alg, dev), groups)
                    if math.isfinite(t) and t > 0:
                        log_n.append(math.log(float(N)))
                        log_t.append(math.log(t))
                if len(log_n) >= 2:
                    xbar = sum(log_n) / len(log_n)
                    ybar = sum(log_t) / len(log_t)
                    num = sum((x - xbar) * (y - ybar) for x, y in zip(log_n, log_t))
                    den = sum((x - xbar) ** 2 for x in log_n)
                    slope = num / den if den > 0 else float("nan")
                else:
                    slope = float("nan")
                slopes[(alg, dev)] = slope
        return slopes
