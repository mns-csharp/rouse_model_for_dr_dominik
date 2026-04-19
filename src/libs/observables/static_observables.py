"""StaticObservables — per-chain R², Rg², end-to-end vector, CoM, middle-segment position."""

import torch

from rouse_model_python.src.libs.number_space.number_space import NumberSpace


class StaticObservables:
    @staticmethod
    def compute_R2(positions: torch.Tensor, ns: NumberSpace) -> torch.Tensor:
        unwrapped = ns.unwrap_chains(positions)
        diff = unwrapped[:, -1, :] - unwrapped[:, 0, :]
        return (diff * diff).sum(dim=-1)

    @staticmethod
    def compute_Rg2(positions: torch.Tensor, ns: NumberSpace) -> torch.Tensor:
        unwrapped = ns.unwrap_chains(positions)
        cm = unwrapped.mean(dim=1, keepdim=True)
        diff = unwrapped - cm
        return (diff * diff).sum(dim=2).mean(dim=1)

    @staticmethod
    def compute_end_to_end_vector(positions: torch.Tensor,
                                  ns: NumberSpace) -> torch.Tensor:
        unwrapped = ns.unwrap_chains(positions)
        return unwrapped[:, -1, :] - unwrapped[:, 0, :]

    @staticmethod
    def compute_center_of_mass(positions: torch.Tensor,
                               ns: NumberSpace) -> torch.Tensor:
        unwrapped = ns.unwrap_chains(positions)
        return unwrapped.mean(dim=1)

    @staticmethod
    def compute_middle_segment_position(positions: torch.Tensor,
                                        ns: NumberSpace) -> torch.Tensor:
        unwrapped = ns.unwrap_chains(positions)
        N = positions.shape[1]
        mid = N // 2
        return unwrapped[:, mid, :].clone()
