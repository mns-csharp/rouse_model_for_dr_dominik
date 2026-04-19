"""SimWorkItem — describes one simulation for cost-aware device assignment."""

from dataclasses import dataclass


@dataclass
class SimWorkItem:
    index: int
    N: int
    n_chains: int
    sweeps: int
    cost: float = 0.0
    device: str = ""
    sub_batches: int = 1
