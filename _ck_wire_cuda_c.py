"""Bulk-wire CHECKLIST_COUNTERS into mc.py, rescale-state into simulation.py,
and probe calls into main.py for the remaining 3 GPU cuda_c apps.

Pattern matches the reference cuda_c app already wired
(g1c_cc_gpu_single_thread_conventional_mc_cuda_c).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


APPS = [
    {
        "name": "gnc_cc_gpu_multi_thread_conventional_mc_cuda_c",
        "logger_tag": "INDEP-gpu-multi-conv-cudac",
        "uses_streams": True,
        "uses_batch_size": False,
    },
    {
        "name": "g1m_cc_gpu_single_thread_multistep_mc_cuda_c",
        "logger_tag": "INDEP-gpu-single-multi-cudac",
        "uses_streams": False,
        "uses_batch_size": True,
    },
    {
        "name": "gnm_cc_gpu_multi_thread_multistep_mc_cuda_c",
        "logger_tag": "INDEP-gpu-multi-multi-cudac",
        "uses_streams": True,
        "uses_batch_size": True,
    },
]


COUNTER_DECL = """

CHECKLIST_COUNTERS = {
    "ev_calls": 0,
    "max_disp": 0.0,
    "disp_violations": 0,
}


def _reset_counters():
    CHECKLIST_COUNTERS["ev_calls"] = 0
    CHECKLIST_COUNTERS["max_disp"] = 0.0
    CHECKLIST_COUNTERS["disp_violations"] = 0
"""


def patch_mc(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "CHECKLIST_COUNTERS" in src:
        return False
    # Insert after _MOVE_NAMES line.
    src = src.replace(
        '_MOVE_NAMES = ("hinge", "n_tail", "c_tail")',
        '_MOVE_NAMES = ("hinge", "n_tail", "c_tail")\n' + COUNTER_DECL.lstrip(),
        1,
    )
    # Add ev_calls + max_disp tracking right after each _fused_accept call (multistep)
    # or right after the propose+delta_e block (conventional). We piggyback on
    # the line `accepted = ` in multistep, or `accept = rng.random() < prob` in
    # conventional single-stream. For multi-thread conventional this is the
    # `_fused_accept` host loop in the round dispatcher.
    if "accepted = _fused_accept(" in src:
        src = src.replace(
            "accepted = _fused_accept(de_np, corr_np, nm_np, kBT, u_np)",
            (
                "accepted = _fused_accept(de_np, corr_np, nm_np, kBT, u_np)\n\n"
                "        CHECKLIST_COUNTERS[\"ev_calls\"] += int(B)\n"
                "        diffs = (new_ca - old_ca)\n"
                "        diffs = diffs - cfg.box_size * torch.round(diffs / cfg.box_size)\n"
                "        disps_all = diffs.norm(dim=-1)\n"
                "        if int(disps_all.numel()) > 0:\n"
                "            d_max = float(disps_all.max().item())\n"
                "            if d_max > CHECKLIST_COUNTERS[\"max_disp\"]:\n"
                "                CHECKLIST_COUNTERS[\"max_disp\"] = d_max\n"
                "            n_viol = int((disps_all > cfg.l0 * 1.05).sum().item())\n"
                "            CHECKLIST_COUNTERS[\"disp_violations\"] += n_viol"
            ),
            1,
        )
    elif "accept = rng.random() < prob" in src:
        src = src.replace(
            "accept = rng.random() < prob",
            (
                "CHECKLIST_COUNTERS[\"ev_calls\"] += 1\n"
                "        diff = (new_ca[0, :nm] - old_ca[0, :nm])\n"
                "        diff = diff - box * torch.round(diff / box)\n"
                "        disps = diff.norm(dim=-1)\n"
                "        d_max = float(disps.max().item())\n"
                "        if d_max > CHECKLIST_COUNTERS[\"max_disp\"]:\n"
                "            CHECKLIST_COUNTERS[\"max_disp\"] = d_max\n"
                "        n_viol = int((disps > cfg.l0 * 1.05).sum().item())\n"
                "        CHECKLIST_COUNTERS[\"disp_violations\"] += n_viol\n"
                "        accept = rng.random() < prob"
            ),
            1,
        )
    path.write_text(src, encoding="utf-8")
    return True


RESCALE_NEW = """    def _maybe_rescale(self, sweep_idx: int) -> None:
        cfg = self.cfg
        if cfg.n_small_steps > 0 and (sweep_idx + 1) % cfg.n_small_steps == 0:
            drift, pre = self.state.rescale_bonds()
            rs = getattr(cfg, "_rescale_state", None)
            if rs is None:
                rs = {"count": 0, "max_drift": 0.0, "last_drift": 0.0}
                cfg._rescale_state = rs
            rs["count"] += 1
            rs["last_drift"] = float(drift)
            if drift > rs["max_drift"]:
                rs["max_drift"] = float(drift)
            ss = getattr(cfg, "_stretch_state", None)
            if ss is None:
                ss = {"max": 0.0}
                cfg._stretch_state = ss
            if pre > ss["max"]:
                ss["max"] = float(pre)
            logger.info('bond rescale @ sweep %d: max_stretch_pre=%.5f', sweep_idx + 1, pre)
            if pre > self.cfg.bond_stretch_warn:
                logger.warning('bond stretch %.4f exceeds warn %.4f', pre, self.cfg.bond_stretch_warn)
            if pre > cfg.bond_stretch_raise:
                raise RuntimeError(f"bond stretch {pre:.4f} > raise threshold")"""


def patch_simulation(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "_rescale_state" in src:
        return False
    pattern = re.compile(
        r"    def _maybe_rescale\(self, sweep_idx: int\) -> None:\n"
        r"        cfg = self\.cfg\n"
        r"        if cfg\.n_small_steps > 0 and \(sweep_idx \+ 1\) % cfg\.n_small_steps == 0:\n"
        r"            _, pre = self\.state\.rescale_bonds\(\)\n"
        r"            logger\.info\('bond rescale @ sweep %d: max_stretch_pre=%\.5f', sweep_idx \+ 1, pre\)\n"
        r"            if pre > self\.cfg\.bond_stretch_warn:\n"
        r"                logger\.warning\('bond stretch %\.4f exceeds warn %\.4f', pre, self\.cfg\.bond_stretch_warn\)\n"
        r"            if pre > cfg\.bond_stretch_raise:\n"
        r"                raise RuntimeError\(f\"bond stretch \{pre:\.4f\} > raise threshold\"\)"
    )
    new_src = pattern.sub(RESCALE_NEW, src)
    if new_src == src:
        # Try variant with two-line raise.
        pattern2 = re.compile(
            r"    def _maybe_rescale\(self, sweep_idx: int\) -> None:\n"
            r"        cfg = self\.cfg\n"
            r"        if cfg\.n_small_steps > 0 and \(sweep_idx \+ 1\) % cfg\.n_small_steps == 0:\n"
            r"            _, pre = self\.state\.rescale_bonds\(\)\n"
            r"            logger\.info\('bond rescale @ sweep %d: max_stretch_pre=%\.5f', sweep_idx \+ 1, pre\)\n"
            r"            if pre > self\.cfg\.bond_stretch_warn:\n"
            r"                logger\.warning\('bond stretch %\.4f exceeds warn %\.4f', pre, self\.cfg\.bond_stretch_warn\)\n"
            r"            if pre > cfg\.bond_stretch_raise:\n"
            r"                raise RuntimeError\(\n"
            r"                    f\"bond stretch \{pre:\.4f\} > raise threshold\"\)"
        )
        new_src = pattern2.sub(RESCALE_NEW, src)
    if new_src == src:
        print(f"WARN: simulation.py pattern not matched for {path}")
        return False
    path.write_text(new_src, encoding="utf-8")
    return True


def patch_main(path: Path, app_name: str, logger_tag: str) -> bool:
    src = path.read_text(encoding="utf-8")
    if "_checklist" in src:
        return False
    # Add imports.
    src = src.replace(
        "from . import io_utils\nfrom .logging_setup import configure_logging",
        "from . import io_utils\nfrom . import _checklist\nfrom . import mc as _mc_mod\nfrom .logging_setup import configure_logging",
    )
    # Add cap flags before args = p.parse_args(argv)
    src = src.replace(
        '    args = p.parse_args(argv)',
        '    p.add_argument("--cap_inner_hinge", action="store_true")\n'
        '    p.add_argument("--cap_tail", action="store_true")\n'
        '    args = p.parse_args(argv)',
        1,
    )
    # Add cap config setters before "return cfg"
    src = src.replace(
        "    return cfg",
        "    cfg.cap_inner_hinge = args.cap_inner_hinge\n    cfg.cap_tail = args.cap_tail\n    return cfg",
        1,
    )
    # Wrap main() body — replace from "log = configure_logging" through "return 0".
    pat = re.compile(
        r"(    log = configure_logging\([^)]+\))\s*\n"
        r"    sim = Simulation\(cfg\)\n"
        r"    sim\.initialize\(\)\n"
        r"\n?"
        r"    eq = sim\.run_equilibration\(\s*\n?\s*log_path=([^)]+?)\)\s*\n"
        r"    print\(f\"\[" + re.escape(logger_tag) + r"\] eq done in \{eq\['wall_s'\]:\.2f\}s \"\s*\n"
        r"          f\"\(\{eq\['n_sweeps'\]/max\(eq\['wall_s'\],1e-12\):\.2f\} sweeps/s\)\", flush=True\)\s*\n"
        r"    prod = sim\.run_production\([^)]*\)\s*\n"
        r"    print\(f\"\[" + re.escape(logger_tag) + r"\] prod done in \{prod\['wall_s'\]:\.2f\}s \"\s*\n"
        r"          f\"\(\{prod\['n_sweeps'\]/max\(prod\['wall_s'\],1e-12\):\.2f\} sweeps/s\)\", flush=True\)\s*\n"
        r"    sim\.write_summary\(os\.path\.join\(out, \"summary\.json\"\), eq, prod\)\s*\n"
        r"    return 0",
        re.DOTALL,
    )
    replacement = (
        r"\1\n\n"
        f"    app_name = \"{app_name}\"\n"
        "    _checklist.emit_static(cfg, app_name=app_name)\n\n"
        "    _mc_mod._reset_counters()\n"
        "    sim = Simulation(cfg)\n"
        "    sim.initialize()\n"
        "    _checklist.emit_post_init(sim.state, cfg)\n"
        "    _checklist.emit_synthetic(cfg, sim.rng)\n\n"
        "    traj_dir = io_utils.ensure_dir(os.path.join(out, \"trajectory\")) \\\n"
        "        if getattr(cfg, \"traj_stride\", 0) > 0 else None\n\n"
        "    _checklist.emit_phase_boundary(\"eq\")\n"
        "    eq = sim.run_equilibration(log_path=os.path.join(out, \"eq_observables.tsv\"))\n"
        "    _checklist.emit_post_eq(eq, cfg)\n"
        f"    print(f\"[{logger_tag}] eq done in {{eq['wall_s']:.2f}}s \"\n"
        "          f\"({eq['n_sweeps']/max(eq['wall_s'],1e-12):.2f} sweeps/s)\", flush=True)\n\n"
        "    _checklist.emit_phase_boundary(\"prod\")\n"
        "    prod = sim.run_production(log_path=os.path.join(out, \"prod_observables.tsv\"),\n"
        "                              traj_dir=traj_dir,\n"
        "                              traj_stride=getattr(cfg, \"traj_stride\", 0))\n"
        f"    print(f\"[{logger_tag}] prod done in {{prod['wall_s']:.2f}}s \"\n"
        "          f\"({prod['n_sweeps']/max(prod['wall_s'],1e-12):.2f} sweeps/s)\", flush=True)\n\n"
        "    summary_path = os.path.join(out, \"summary.json\")\n"
        "    sim.write_summary(summary_path, eq, prod)\n\n"
        "    _checklist.emit_post_run(prod, traj_dir, cfg, summary_path,\n"
        "                             sim.stats, sim.state, eq, app_name)\n"
        "    return 0"
    )
    new_src, count = pat.subn(replacement, src)
    if count == 0:
        print(f"WARN: main.py pattern not matched for {path}; manual edit needed")
        return False
    path.write_text(new_src, encoding="utf-8")
    return True


def main():
    for app in APPS:
        d = SRC / app["name"]
        ok_mc = patch_mc(d / "mc.py")
        ok_sim = patch_simulation(d / "simulation.py")
        ok_main = patch_main(d / "main.py", app["name"], app["logger_tag"])
        print(f"{app['name']}: mc={ok_mc} sim={ok_sim} main={ok_main}")


if __name__ == "__main__":
    main()
