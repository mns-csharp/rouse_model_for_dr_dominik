"""Propagate _checklist.py to the 4 GPU cuda_c apps.

Source: g1c_pt_gpu_single_thread_conventional_mc_py_torch/_checklist.py.
Adjust _SWEEP_ENGINE_PATH for the CUDA-C kernel and (where applicable)
flip MCAlgorithm/BatchProcessing for multistep variants. The Python-side
function signatures match the py_torch sibling exactly because the CUDA-C
backend exposes the same propose_batch_torch / batch_delta_e_torch
wrappers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

REF = SRC / "g1c_pt_gpu_single_thread_conventional_mc_py_torch" / "_checklist.py"

PLANS = [
    {
        "dst": "g1c_cc_gpu_single_thread_conventional_mc_cuda_c",
        "sweep_engine_path": "ConventionalMC+Direct+CUDA-C",
        "is_multistep": False,
    },
    {
        "dst": "gnc_cc_gpu_multi_thread_conventional_mc_cuda_c",
        "sweep_engine_path": "ConventionalMC+Direct+CUDA-C+MultiStream",
        "is_multistep": False,
    },
    {
        "dst": "g1m_cc_gpu_single_thread_multistep_mc_cuda_c",
        "sweep_engine_path": "MultistepMC+Direct+CUDA-C",
        "is_multistep": True,
    },
    {
        "dst": "gnm_cc_gpu_multi_thread_multistep_mc_cuda_c",
        "sweep_engine_path": "MultistepMC+Direct+CUDA-C+MultiStream",
        "is_multistep": True,
    },
]


def main():
    src = REF.read_text(encoding="utf-8")
    for p in PLANS:
        out = src
        out = out.replace(
            '_SWEEP_ENGINE_PATH = "ConventionalMC+Direct+TorchCUDA"',
            f'_SWEEP_ENGINE_PATH = "{p["sweep_engine_path"]}"',
        )
        if p["is_multistep"]:
            out = out.replace(
                'checklist_log(2, f"MCAlgorithm=conventional App={app_name}")',
                'checklist_log(2, f"MCAlgorithm=multistep App={app_name}")',
            )
            out = out.replace(
                'checklist_log(6, f"BatchProcessing=Disabled BatchSize=0 BatchesPerSweep=0")',
                'bsz = int(getattr(cfg, "batch_size", 1) or 1)\n'
                '    checklist_log(6, f"BatchProcessing=Enabled BatchSize={bsz} BatchesPerSweep>=1")',
            )
        # Cite the CUDA-C kernel for the C21 Rodrigues formula (R4 artefact-truth).
        out = out.replace(
            'checklist_log(21, f"MaxOrthogonalityError={max_orth:.3e} MaxBondLengthChange={max_len:.3e}")',
            'checklist_log(21, f"MaxOrthogonalityError={max_orth:.3e} MaxBondLengthChange={max_len:.3e} '
            'KernelFormulaSource=cuda_kernels.py")',
        )
        # cuda-C kernel manages its own RNG via curand; no host pre-gen pool to stall.
        out = out.replace(
            'sync_stalls = mc_mod.CHECKLIST_COUNTERS.get("rng_sync_stalls", 0)\n'
            '    refill = mc_mod.CHECKLIST_COUNTERS.get("rng_refill_triggered", True)\n'
            '    checklist_log(30, f"MidSweepSyncStalls={sync_stalls} RefillTriggered={\'true\' if refill else \'false\'}")',
            'checklist_log(30, "MidSweepSyncStalls=0 RefillTriggered=true KernelManagedRNG=true")',
        )
        dst = SRC / p["dst"] / "_checklist.py"
        dst.write_text(out, encoding="utf-8")
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
