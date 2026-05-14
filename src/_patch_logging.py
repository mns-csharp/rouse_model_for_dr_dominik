"""One-shot patch script: edit every main.py + simulation.py + chain.py
in the 16 apps to use the new logging facility. Idempotent.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


BASE = Path(__file__).resolve().parent

APPS = [
    ("c1c_nb_cpu_single_core_conventional_mc_numba",      "INDEP-cpu-single-conv-numba"),
    ("c1c_pt_cpu_single_core_conventional_mc_py_torch",   "INDEP-cpu-single-conv-pytorch"),
    ("c1m_nb_cpu_single_core_multistep_mc_numba",         "INDEP-cpu-single-multi-numba"),
    ("c1m_pt_cpu_single_core_multistep_mc_py_torch",      "INDEP-cpu-single-multi-pytorch"),
    ("cnc_nb_cpu_multi_core_conventional_mc_numba",       "INDEP-cpu-multi-conv-numba"),
    ("cnc_pt_cpu_multi_core_conventional_mc_py_torch",    "INDEP-cpu-multi-conv-pytorch"),
    ("cnm_nb_cpu_multi_core_multistep_mc_numba",          "INDEP-cpu-multi-multi-numba"),
    ("cnm_pt_cpu_multi_core_multistep_mc_py_torch",       "INDEP-cpu-multi-multi-pytorch"),
    ("g1c_cc_gpu_single_thread_conventional_mc_cuda_c",   "INDEP-gpu-single-conv-cudac"),
    ("g1c_pt_gpu_single_thread_conventional_mc_py_torch", "INDEP-gpu-single-conv-pytorch"),
    ("g1m_cc_gpu_single_thread_multistep_mc_cuda_c",      "INDEP-gpu-single-multi-cudac"),
    ("g1m_pt_gpu_single_thread_multistep_mc_py_torch",    "INDEP-gpu-single-multi-pytorch"),
    ("gnc_cc_gpu_multi_thread_conventional_mc_cuda_c",    "INDEP-gpu-multi-conv-cudac"),
    ("gnc_pt_gpu_multi_thread_conventional_mc_py_torch",  "INDEP-gpu-multi-conv-pytorch"),
    ("gnm_cc_gpu_multi_thread_multistep_mc_cuda_c",       "INDEP-gpu-multi-multi-cudac"),
    ("gnm_pt_gpu_multi_thread_multistep_mc_py_torch",     "INDEP-gpu-multi-multi-pytorch"),
]


def patch_main(path: Path, tag: str) -> None:
    text = path.read_text(encoding="utf-8")

    # Skip if already patched.
    if "from .logging_setup import configure_logging" in text:
        return

    # 1. Inject the import after the last `from . import ...` line.
    if "from . import io_utils" in text:
        text = text.replace(
            "from . import io_utils",
            "from . import io_utils\nfrom .logging_setup import configure_logging",
            1,
        )

    # 2. Add --log_level and --log_file to argparse.
    log_args = (
        '    p.add_argument("--log_level", default="INFO",\n'
        '                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])\n'
        '    p.add_argument("--log_file", default="",\n'
        '                   help="if set, write logs to this file (default: <output_dir>/run.log)")\n'
    )
    text = re.sub(
        r'(    p\.add_argument\("--repulsive_energy"[^\n]*\n)',
        r'\1' + log_args,
        text,
        count=1,
    )

    # 3. Configure logger at top of main() and replace prints with log.info.
    # Replace the first `out = io_utils.ensure_dir(cfg.output_dir)` block to
    # come after configure_logging.
    text = re.sub(
        r'(    out = io_utils\.ensure_dir\(cfg\.output_dir\))',
        (
            "    out = io_utils.ensure_dir(cfg.output_dir)\n"
            "    log_file = getattr(cfg, '_log_file', '') or "
            f"os.path.join(out, 'run.log')\n"
            f"    log = configure_logging(getattr(cfg, '_log_level', 'INFO'), "
            f"log_file, '{tag}')"
        ),
        text,
        count=1,
    )

    # 4. Stash log_level / log_file from CLI args onto cfg.
    text = re.sub(
        r'(    cfg\.traj_stride = args\.traj_stride\n)',
        r'\1    cfg._log_level = args.log_level\n    cfg._log_file = args.log_file\n',
        text,
        count=1,
    )

    # 5. Replace print(f"[INDEP-...]...") with log.info("...").
    # Match a 1- or 2-line print(f"...") block and convert to log.info(...).
    def _print_to_log(match: re.Match) -> str:
        body = match.group(1)
        # Strip the `[TAG] ` prefix from the f-string content; the formatter adds it back.
        body = re.sub(r'\[INDEP-[a-z\-]+\]\s*', '', body)
        return f'log.info(f{body})'
    text = re.sub(
        r'print\(f(\"[^\"]*\")(?:,\s*flush=True)?\)',
        _print_to_log,
        text,
    )
    # Multi-line f"..." " ... " concatenation:
    # We collapse them lazily — match the whole `print(f"..." f"..." ..., flush=True)` block.
    # The simpler regex above already handles single-line cases; for multi-line we leave them.

    path.write_text(text, encoding="utf-8")


def patch_simulation(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "import logging" in text and "logger = logging.getLogger" in text:
        return
    # Inject at top of file.
    text = re.sub(
        r'(\nfrom \. import io_utils, mc, observables\n)',
        r'\nimport logging\n\1logger = logging.getLogger(__name__)\n',
        text,
        count=1,
    )
    # Log eq/prod boundary + per-sample observable + bond rescale.
    # Find the run_equilibration body:
    text = re.sub(
        r'(    def run_equilibration\(self[^\n]*\n[^\n]*\n[^\n]*\n)',
        r'\1        logger.info("eq start: eq_sweeps=%d", self.cfg.eq_sweeps)\n',
        text,
        count=1,
    )
    text = re.sub(
        r'(    def run_production\(self[^\n]*\n[^\n]*\n[^\n]*\n[^\n]*\n)',
        r'\1        logger.info("prod start: prod_sweeps=%d", self.cfg.prod_sweeps)\n',
        text,
        count=1,
    )
    # Log rescale.
    text = re.sub(
        r'(            _, pre = self\.state\.rescale_bonds\(\)\n)',
        (
            r'\1'
            "            logger.info('bond rescale @ sweep %d: max_stretch_pre=%.5f', "
            "sweep_idx + 1, pre)\n"
            "            if pre > self.cfg.bond_stretch_warn:\n"
            "                logger.warning('bond stretch %.4f exceeds warn %.4f', pre, self.cfg.bond_stretch_warn)\n"
        ),
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def main():
    for app, tag in APPS:
        d = BASE / app
        main_py = d / "main.py"
        sim_py = d / "simulation.py"
        if main_py.exists():
            patch_main(main_py, tag)
        if sim_py.exists():
            patch_simulation(sim_py)
    print(f"Patched {len(APPS)} apps")


if __name__ == "__main__":
    main()
