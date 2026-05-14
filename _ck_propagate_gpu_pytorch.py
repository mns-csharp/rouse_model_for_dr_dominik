"""Propagate _checklist.py from g1c_pt_gpu_single_thread_conventional_mc_py_torch to:
- gnc_pt_gpu_multi_thread_conventional_mc_py_torch
- g1m_pt_gpu_single_thread_multistep_mc_py_torch
- gnm_pt_gpu_multi_thread_multistep_mc_py_torch

Adjust per-cluster constants and (for multistep) the engine batch_delta_e
shape. Engine wiring (mc.py / simulation.py / main.py) is done manually
since each main.py has slightly different argparse surfaces."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

REF = SRC / "g1c_pt_gpu_single_thread_conventional_mc_py_torch" / "_checklist.py"

PLANS = [
    {
        "dst": "gnc_pt_gpu_multi_thread_conventional_mc_py_torch",
        "sweep_engine_path": "ConventionalMC+Direct+TorchCUDA+MultiStream",
        "is_multistep": False,
    },
    {
        "dst": "g1m_pt_gpu_single_thread_multistep_mc_py_torch",
        "sweep_engine_path": "MultistepMC+Direct+TorchCUDA",
        "is_multistep": True,
    },
    {
        "dst": "gnm_pt_gpu_multi_thread_multistep_mc_py_torch",
        "sweep_engine_path": "MultistepMC+Direct+TorchCUDA+MultiStream",
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
        dst = SRC / p["dst"] / "_checklist.py"
        dst.write_text(out, encoding="utf-8")
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
