"""EnergyKernels — three-zone energy kernel + batched delta-E and EMM correction.

All methods are @staticmethod. Includes torch.compile-wrapped GPU kernels
(compiled lazily on first access to avoid module-level side effects).
"""

import torch

from rouse_model_python.src.libs.number_space.number_space import NumberSpace


class EnergyKernels:
    _COMPILE_MODE = 'default'
    _compiled_delta_e = None
    _compiled_emm_corr = None

    @staticmethod
    def _triton_available() -> bool:
        try:
            import triton  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def energy_kernel(r2: torch.Tensor, r_rep_sq: float, r_max_sq: float,
                      repulsive_energy: float, contact_energy: float) -> torch.Tensor:
        e = torch.zeros_like(r2)
        repulsive_mask = r2 < r_rep_sq
        e[repulsive_mask] = repulsive_energy
        if contact_energy != 0.0:
            contact_mask = (r2 >= r_rep_sq) & (r2 < r_max_sq)
            e[contact_mask] = contact_energy
        return e

    @staticmethod
    def compute_segment_pair_energy(pos_a: torch.Tensor, pos_b: torch.Tensor,
                                    ns: NumberSpace, r_rep_sq: float,
                                    repulsive_energy: float) -> float:
        delta = pos_b.unsqueeze(0) - pos_a.unsqueeze(1)
        delta = delta - ns.box_size * torch.round(delta * ns._inv_box)
        r2 = (delta * delta).sum(dim=2)
        overlap = r2 < r_rep_sq
        if not overlap.any():
            return 0.0
        return overlap.sum().item() * repulsive_energy

    @staticmethod
    def _batched_delta_e_impl(pos_f32, old_batch, new_batch,
                              gs_tensor, nm_tensor,
                              box, inv_box, r_rep_sq, rep_e,
                              V, max_moved, n_total):
        d_old = pos_f32[None, None, :, :] - old_batch[:, :, None, :]
        d_old = d_old - box * torch.round(d_old * inv_box)
        r2_old = (d_old * d_old).sum(dim=3)
        d_new = pos_f32[None, None, :, :] - new_batch[:, :, None, :]
        d_new = d_new - box * torch.round(d_new * inv_box)
        r2_new = (d_new * d_new).sum(dim=3)
        bead_idx = torch.arange(n_total, device=pos_f32.device).unsqueeze(0)
        self_mask = (bead_idx >= gs_tensor.unsqueeze(1)) & \
                    (bead_idx < (gs_tensor + nm_tensor).unsqueeze(1))
        self_mask_3d = self_mask.unsqueeze(1)
        r2_old = r2_old.masked_fill(self_mask_3d, float('inf'))
        r2_new = r2_new.masked_fill(self_mask_3d, float('inf'))
        move_idx = torch.arange(max_moved, device=pos_f32.device).unsqueeze(0)
        pad_mask = move_idx >= nm_tensor.unsqueeze(1)
        pad_mask_3d = pad_mask.unsqueeze(2)
        r2_old = r2_old.masked_fill(pad_mask_3d, float('inf'))
        r2_new = r2_new.masked_fill(pad_mask_3d, float('inf'))
        e_old_v = (r2_old < r_rep_sq).reshape(V, -1).sum(dim=1)
        e_new_v = (r2_new < r_rep_sq).reshape(V, -1).sum(dim=1)
        delta_v = (e_new_v.float() - e_old_v.float()) * rep_e
        return delta_v

    @staticmethod
    def _batched_emm_correction_impl(old_batch, new_batch,
                                     pair_i, pair_j,
                                     box, inv_box, r_rep_sq, rep_e, B):
        n_pairs = pair_i.shape[0]
        if n_pairs == 0:
            return torch.zeros(B, B, dtype=torch.float32, device=old_batch.device)
        oi = old_batch[pair_i]
        ni = new_batch[pair_i]
        oj = old_batch[pair_j]
        nj = new_batch[pair_j]
        oi_e = oi[:, :, None, :]
        ni_e = ni[:, :, None, :]
        oj_e = oj[:, None, :, :]
        nj_e = nj[:, None, :, :]
        d = oi_e - oj_e
        d = d - box * torch.round(d * inv_box)
        c00 = ((d * d).sum(-1) < r_rep_sq).sum(dim=(1, 2))
        d = oi_e - nj_e
        d = d - box * torch.round(d * inv_box)
        c01 = ((d * d).sum(-1) < r_rep_sq).sum(dim=(1, 2))
        d = ni_e - oj_e
        d = d - box * torch.round(d * inv_box)
        c10 = ((d * d).sum(-1) < r_rep_sq).sum(dim=(1, 2))
        d = ni_e - nj_e
        d = d - box * torch.round(d * inv_box)
        c11 = ((d * d).sum(-1) < r_rep_sq).sum(dim=(1, 2))
        corr_vals = ((c11.float() - c01.float()) - (c10.float() - c00.float())) * rep_e
        corr = torch.zeros(B, B, dtype=torch.float32, device=old_batch.device)
        corr[pair_i, pair_j] = corr_vals
        return corr

    @classmethod
    def delta_e_kernel(cls):
        if cls._compiled_delta_e is None:
            if cls._triton_available():
                try:
                    cls._compiled_delta_e = torch.compile(
                        cls._batched_delta_e_impl,
                        mode=cls._COMPILE_MODE, dynamic=True)
                except Exception:
                    cls._compiled_delta_e = cls._batched_delta_e_impl
            else:
                cls._compiled_delta_e = cls._batched_delta_e_impl
        return cls._compiled_delta_e

    @classmethod
    def emm_correction_kernel(cls):
        if cls._compiled_emm_corr is None:
            if cls._triton_available():
                try:
                    cls._compiled_emm_corr = torch.compile(
                        cls._batched_emm_correction_impl,
                        mode=cls._COMPILE_MODE, dynamic=True)
                except Exception:
                    cls._compiled_emm_corr = cls._batched_emm_correction_impl
            else:
                cls._compiled_emm_corr = cls._batched_emm_correction_impl
        return cls._compiled_emm_corr
