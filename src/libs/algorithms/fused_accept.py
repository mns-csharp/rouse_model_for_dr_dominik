"""fused_accept — Numba JIT for the sequential accept + rank-1 correction loop
that dominates the Python side of MultiStepMC._process_batch.

Semantics match the original Python loop in `multistep_mc.py::_process_batch`:
  for i in range(B):
    accepted[i] = metropolis(delta_e[i], uniform[i])
    if accepted[i]:
      for j in range(i+1, B):
        delta_e[j] += correction[i, j]

RNG draws: caller must pre-draw B uniforms in the exact same order the
original Python loop would have consumed them (one per proposal with
n_moved > 0), so the acceptance outcomes match bit-for-bit given the same
seed.

`n_moved` lets us skip proposals that were empty (n_moved == 0) — the
original Metropolis call never fires for those, so `accepted[i]` stays False
and no rank-1 update runs. The caller must still consume a uniform for
empty proposals if it wants to preserve the RNG stream; for simplicity we
consume zero uniforms per empty slot here and the caller must match.

`fused_accept_and_correct_torch` is the on-GPU variant used when
`cfg.accept_on_gpu` is set — same semantics, runs as a single-thread CuPy
RawKernel that shares memory with torch tensors via DLPack. Eliminates the
per-batch blocking D2H of delta_e / correction / uniforms in
`MultiStepMC._process_batch`.
"""

import math
import numba
import numpy as np
import torch


@numba.njit(cache=True, fastmath=False)
def fused_accept_and_correct(delta_e: np.ndarray,
                             correction: np.ndarray,
                             n_moved: np.ndarray,
                             kBT: float,
                             uniforms: np.ndarray) -> np.ndarray:
    """Metropolis-accept with sequential rank-1 update in native code.

    Args:
      delta_e:    float32[B]    per-proposal delta-E; mutated in-place.
      correction: float32[B,B]  upper-triangular rank-1 correction matrix.
      n_moved:    int64[B]      per-proposal bead count; 0 means skip.
      kBT:        float         Boltzmann factor.
      uniforms:   float32[B]    pre-drawn U(0,1) samples (one per proposal).

    Returns:
      accepted: bool[B]
    """
    B = delta_e.shape[0]
    accepted = np.zeros(B, dtype=np.bool_)
    u_idx = 0
    for i in range(B):
        if n_moved[i] == 0:
            continue
        de = delta_e[i]
        if de <= 0.0:
            prob = 1.0
        else:
            exponent = -de / kBT
            if exponent <= -745.0:
                prob = 0.0
            elif exponent >= 709.0:
                prob = 1.0
            else:
                prob = math.exp(exponent)
        if uniforms[u_idx] < prob:
            accepted[i] = True
            for j in range(i + 1, B):
                if n_moved[j] == 0:
                    continue
                c = correction[i, j]
                if c != 0.0:
                    delta_e[j] += c
        u_idx += 1
    return accepted


_FUSED_ACCEPT_CUDA_SRC = r"""
extern "C" __global__
void fused_accept_kernel(
    float* __restrict__ delta_e,
    const float* __restrict__ correction,
    const long long* __restrict__ n_moved,
    const float* __restrict__ uniforms,
    unsigned char* __restrict__ accepted,
    int B,
    double kBT
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int u_idx = 0;
    for (int i = 0; i < B; ++i) {
        if (n_moved[i] == 0) continue;
        double de = (double)delta_e[i];
        double prob;
        if (de <= 0.0) {
            prob = 1.0;
        } else {
            double exponent = -de / kBT;
            if (exponent <= -745.0) {
                prob = 0.0;
            } else if (exponent >= 709.0) {
                prob = 1.0;
            } else {
                prob = exp(exponent);
            }
        }
        double u = (double)uniforms[u_idx];
        if (u < prob) {
            accepted[i] = 1;
            for (int j = i + 1; j < B; ++j) {
                if (n_moved[j] == 0) continue;
                float c = correction[i * B + j];
                if (c != 0.0f) {
                    delta_e[j] = delta_e[j] + c;
                }
            }
        }
        ++u_idx;
    }
}
"""

_fused_accept_kernel_cached = None


def _get_fused_accept_kernel():
    global _fused_accept_kernel_cached
    if _fused_accept_kernel_cached is None:
        import cupy as cp
        _fused_accept_kernel_cached = cp.RawKernel(
            _FUSED_ACCEPT_CUDA_SRC, "fused_accept_kernel")
    return _fused_accept_kernel_cached


def fused_accept_and_correct_torch(
    delta_e: torch.Tensor,
    correction: torch.Tensor,
    n_moved: torch.Tensor,
    kBT: float,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """On-GPU sequential Metropolis accept + rank-1 delta_e correction.

    Bit-identical to fused_accept_and_correct given the same inputs and
    uniforms stream. No host-device sync — all tensors stay on device.

    Args:
      delta_e:    float32[B], CUDA    per-proposal delta-E; mutated in-place.
      correction: float32[B,B], CUDA  row-major rank-1 correction matrix.
      n_moved:    int64[B], CUDA      per-proposal bead count; 0 means skip.
      kBT:        float               Boltzmann factor.
      uniforms:   float32[B], CUDA    pre-drawn U(0,1) samples.

    Returns:
      accepted: bool[B], CUDA.
    """
    import cupy as cp
    assert delta_e.is_cuda and correction.is_cuda and n_moved.is_cuda and uniforms.is_cuda, \
        "fused_accept_and_correct_torch requires CUDA tensors"
    assert delta_e.dtype == torch.float32, f"delta_e must be float32, got {delta_e.dtype}"
    assert correction.dtype == torch.float32, f"correction must be float32, got {correction.dtype}"
    assert uniforms.dtype == torch.float32, f"uniforms must be float32, got {uniforms.dtype}"
    assert n_moved.dtype == torch.int64, f"n_moved must be int64, got {n_moved.dtype}"

    B = delta_e.shape[0]
    assert correction.shape == (B, B), f"correction shape {correction.shape} != ({B},{B})"
    assert n_moved.shape == (B,)
    # uniforms must hold at least one slot per nonzero n_moved entry; we leave
    # that bookkeeping to the caller (matches the numba/numpy contract).

    delta_e = delta_e.contiguous()
    correction = correction.contiguous()
    n_moved = n_moved.contiguous()
    uniforms = uniforms.contiguous()

    accepted = torch.zeros(B, dtype=torch.uint8, device=delta_e.device)

    kernel = _get_fused_accept_kernel()
    de_cp = cp.from_dlpack(delta_e)
    corr_cp = cp.from_dlpack(correction)
    nm_cp = cp.from_dlpack(n_moved)
    u_cp = cp.from_dlpack(uniforms)
    acc_cp = cp.from_dlpack(accepted)

    kernel((1,), (1,),
           (de_cp, corr_cp, nm_cp, u_cp, acc_cp,
            np.int32(B), np.float64(kBT)))
    return accepted.view(torch.bool)
