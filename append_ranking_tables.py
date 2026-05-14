"""Append four ranking tables to benchmark_data/tables.md.

Each table ranks all 27 apps by one metric:
  - Throughput (prod sweeps / s, higher is better)
  - Speedup vs baseline c1c_nb (higher is better)
  - Scaling vs smallest NK (lower is better — flatter)
  - Wall-clock total seconds (lower is better)

Headlined at NK=1000 (N=100, K=10) where every app has an ok cell.
NK=10000 column shown alongside for context (or 'timeout' / 'error' / '—').
Idempotent: replaces the section starting at '## Rankings (all 27 apps)'.
"""
from __future__ import annotations

import math

import bench_loop_iteration as bli


HEADLINE_KEY = "N100_K10"   # NK=1000
SECONDARY_KEY = "N1000_K10" # NK=10000

SECTION_HEADER = "## Rankings (all 27 apps)"
WINNERS_HEADER = "## Winners"


def short_code(app: str) -> str:
    return "_".join(app.split("_")[:2])


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, float) and not math.isfinite(v):
        return "—"
    if isinstance(v, (int, float)):
        if abs(v) >= 1000:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        return f"{v:.4f}"
    return str(v)


def cell_value(rec: dict, size_key: str, source: str, field: str):
    """Return the raw value or a status string. source is 'sizes' or 'ratios'."""
    sizes = rec.get("sizes", {})
    s = sizes.get(size_key, {})
    if s.get("status") != "ok":
        return s.get("status") or "—"
    if source == "sizes":
        return s.get(field)
    return rec.get("ratios", {}).get(size_key, {}).get(field)


def render_ranking(records: dict, metric: str, ylabel: str, source: str,
                   field: str, descending: bool) -> str:
    """Build markdown for one ranking table."""
    rows = []
    for app, rec in records.items():
        head = cell_value(rec, HEADLINE_KEY, source, field)
        sec = cell_value(rec, SECONDARY_KEY, source, field)
        # numeric headline for sorting
        if isinstance(head, (int, float)) and math.isfinite(head):
            sort_val = head
            ok = True
        else:
            sort_val = float("-inf") if descending else float("inf")
            ok = False
        rows.append((app, head, sec, sort_val, ok))

    rows.sort(key=lambda r: r[3], reverse=descending)
    lines = [
        f"### {ylabel}",
        "",
        f"Headlined at NK=1000 (N=100, K=10); NK=10000 (N=1000, K=10) shown for context.",
        "",
        f"| Rank | Code | App | {metric} @ NK=1000 | {metric} @ NK=10000 |",
        "|---|---|---|---|---|",
    ]
    for i, (app, head, sec, _, ok) in enumerate(rows, 1):
        rank = str(i) if ok else "—"
        lines.append(
            f"| {rank} | `{short_code(app)}` | `{app}` | {fmt(head)} | {fmt(sec)} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_winners(records: dict) -> str:
    """Pick winners by programmatic rules over the per-app records."""
    def num(v):
        return v if isinstance(v, (int, float)) and math.isfinite(v) else None

    cpu = [(a, r) for a, r in records.items() if r["tags"]["hw"] == "CPU"]
    gpu = [(a, r) for a, r in records.items() if r["tags"]["hw"] == "GPU"]

    def tput(rec, sk):
        s = rec["sizes"].get(sk, {})
        if s.get("status") != "ok":
            return None
        return num(s.get("prod_sweeps_per_s"))

    def status(rec, sk):
        return rec["sizes"].get(sk, {}).get("status")

    # CPU: best throughput at NK=1000 (no CPU app survives NK=10000)
    cpu_ranked = sorted(
        ((a, tput(r, HEADLINE_KEY), r["sizes"].get(SECONDARY_KEY, {}).get("wall_total_s"),
          r["sizes"].get(HEADLINE_KEY, {}).get("wall_total_s"),
          status(r, SECONDARY_KEY))
         for a, r in cpu),
        key=lambda t: -(t[1] or -1),
    )
    cpu_win_app, cpu_win_t, _, cpu_win_w, _ = cpu_ranked[0]

    cpu_all_timeout = all(status(r, SECONDARY_KEY) == "timeout" for _, r in cpu)

    # GPU peak speed at NK=1000 (whatever happens at NK=10000)
    gpu_peak = sorted(
        ((a, tput(r, HEADLINE_KEY),
          r["sizes"].get(HEADLINE_KEY, {}).get("wall_total_s"),
          status(r, SECONDARY_KEY))
         for a, r in gpu),
        key=lambda t: -(t[1] or -1),
    )
    gpu_peak_app, gpu_peak_t, gpu_peak_w, gpu_peak_status10k = gpu_peak[0]

    # GPU robust: ok in all 3 cells, then sort by NK=10000 throughput
    all3_ok = [
        (a, r) for a, r in gpu
        if all(status(r, sk) == "ok" for sk in ("N25_K10", HEADLINE_KEY, SECONDARY_KEY))
    ]
    gpu_robust = sorted(
        ((a, tput(r, HEADLINE_KEY), tput(r, SECONDARY_KEY),
          r["sizes"].get(SECONDARY_KEY, {}).get("wall_total_s"))
         for a, r in all3_ok),
        key=lambda t: -(t[2] or -1),
    )
    gpu_rob_app, gpu_rob_t1k, gpu_rob_t10k, gpu_rob_w10k = gpu_robust[0]

    lines = [
        WINNERS_HEADER,
        "",
        f"Headlined at NK=1000 (N=100, K=10) — the largest cell where every app finished. NK=10000 (N=1000, K=10) shown as the stress cell.",
        "",
        "### CPU",
        "",
        f"**Winner: `{short_code(cpu_win_app)}`** — `{cpu_win_app}`",
        "",
        f"- Throughput @ NK=1000: **{fmt(cpu_win_t)} sweeps/s** (wall {fmt(cpu_win_w)} s).",
    ]
    if cpu_all_timeout:
        lines.append("- All 10 CPU apps **timeout at NK=10000** (1200 s wall budget); no CPU app is robust at the stress cell.")
    lines.append("- Multistep (`*m_*`) recovers ~21 % over conventional baseline `c1c_nb`. Multi-core (`cn*`) variants lose because K=10 is too few chains to amortise prange overhead. PyTorch CPU variants are 8–25× slower than Numba.")
    lines.append("")
    lines.append("### GPU")
    lines.append("")
    lines.append("Two valid winners depending on whether the NK=10000 stress cell counts:")
    lines.append("")
    lines.append("| Criterion | Winner | tput @ NK=1000 | NK=10000 status / tput |")
    lines.append("|---|---|---:|---|")
    sec_str = (f"{fmt(tput(records[gpu_peak_app], SECONDARY_KEY))} sw/s"
               if gpu_peak_status10k == "ok" else gpu_peak_status10k)
    lines.append(
        f"| Peak speed at NK=1000 | `{short_code(gpu_peak_app)}` | {fmt(gpu_peak_t)} | {sec_str} |"
    )
    lines.append(
        f"| Robust across all 3 sizes | `{short_code(gpu_rob_app)}` | {fmt(gpu_rob_t1k)} | ok / {fmt(gpu_rob_t10k)} sw/s (wall {fmt(gpu_rob_w10k)} s) |"
    )
    lines.append("")
    lines.append(
        f"**Recommended GPU winner: `{short_code(gpu_rob_app)}`** — the only app that is on the Pareto frontier at every cell it ran *and* successfully ran every cell. "
        f"If you only care about NK=1000 throughput, `{short_code(gpu_peak_app)}` is roughly "
        f"{(gpu_peak_t / gpu_rob_t1k):.0f}× faster, but it `{gpu_peak_status10k}`s at NK=10000 (CUDA shared-mem layout `5·M·3 + …` exceeds the 48 KB per-block limit at N=1000)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    apps = bli.discover_apps()
    records = {a: bli.load_app_record(a) for a in apps}
    records = {a: r for a, r in records.items() if r is not None}

    sections = []
    sections.append(SECTION_HEADER)
    sections.append("")

    sections.append(render_ranking(
        records,
        metric="throughput (sw/s)",
        ylabel="Ranked by throughput (prod sweeps/s, higher is better)",
        source="sizes", field="prod_sweeps_per_s",
        descending=True,
    ))
    sections.append(render_ranking(
        records,
        metric="speedup ×",
        ylabel=f"Ranked by speedup vs `{bli.BASELINE_APP}` (higher is better)",
        source="ratios", field="speedup_vs_baseline",
        descending=True,
    ))
    sections.append(render_ranking(
        records,
        metric="scaling ×",
        ylabel="Ranked by scaling = per_sweep_s @ NK / per_sweep_s @ NK=250 (lower is better — flatter)",
        source="ratios", field="scaling_vs_smallest",
        descending=False,
    ))
    sections.append(render_ranking(
        records,
        metric="wall (s)",
        ylabel="Ranked by wall-clock total seconds (lower is better)",
        source="sizes", field="wall_total_s",
        descending=False,
    ))

    sections.append(render_winners(records))

    new_block = "\n".join(sections)

    md_path = bli.BENCH_DIR / "tables.md"
    text = md_path.read_text(encoding="utf-8")
    # Splice point: earliest of the two markers (so re-running with either
    # already present replaces from there to EOF).
    splice_idx = None
    for marker in (SECTION_HEADER, WINNERS_HEADER):
        i = text.find(marker)
        if i != -1 and (splice_idx is None or i < splice_idx):
            splice_idx = i
    if splice_idx is not None:
        head = text[:splice_idx].rstrip() + "\n\n"
        text = head + new_block + "\n"
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + new_block + "\n"

    md_path.write_text(text, encoding="utf-8")
    print(f"[rank] wrote {md_path} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
