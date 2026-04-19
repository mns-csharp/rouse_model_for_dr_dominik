"""GPUKernelManager — compiles and caches all CuPy RawKernels.

CuPy is imported lazily inside __init__ so this module imports cleanly on
machines without CuPy installed. Constructing a GPUKernelManager on such a
host raises ImportError.
"""

from rouse_model_python.src.libs.paths.gpu.gpu_kernel_sources import GPUKernelSources


class GPUKernelManager:
    def __init__(self):
        import cupy as cp
        self._cp = cp
        self._kernels = {}
        self._compiled = False

    def compile_all(self) -> None:
        if self._compiled:
            return
        cp = self._cp
        self._kernels['segment_proposal'] = cp.RawKernel(
            GPUKernelSources.segment_proposal_src(), 'segment_proposal_kernel')
        self._kernels['pivot_proposal'] = cp.RawKernel(
            GPUKernelSources.pivot_proposal_src(), 'pivot_proposal_kernel')
        self._kernels['delta_e'] = cp.RawKernel(
            GPUKernelSources.delta_e_src(), 'delta_e_kernel')
        self._kernels['emm_correction'] = cp.RawKernel(
            GPUKernelSources.emm_correction_src(), 'emm_correction_kernel')
        self._kernels['apply_moves'] = cp.RawKernel(
            GPUKernelSources.apply_moves_src(), 'apply_moves_kernel')
        self._kernels['copy_f64_to_f32'] = cp.RawKernel(
            GPUKernelSources.copy_f64_to_f32_src(), 'copy_f64_to_f32')
        self._compiled = True

    def __getitem__(self, name: str):
        if not self._compiled:
            self.compile_all()
        return self._kernels[name]
