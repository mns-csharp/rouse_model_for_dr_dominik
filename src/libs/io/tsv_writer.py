"""TSVWriter — generic append-row and bench_timings-specific writers."""

import csv
import os


class TSVWriter:
    BENCH_TIMINGS_HEADER = [
        "N", "algorithm", "device", "n_chains", "eq_sweeps", "prod_sweeps",
        "total_wall_s", "per_sweep_s", "hinge_accept", "tail_accept",
        "pivot_accept", "segment_size", "max_angle_hinge", "batch_size",
        "gpu_util_mean", "run_id", "warmup_sweeps", "wall_clock_median_of_3",
        "n_replicas", "per_sweep_s_effective",
    ]

    PE_WINNERS_HEADER = ["N", "residues_per_segment", "max_angle_hinge", "batch_size"]

    @staticmethod
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def append_row(filepath: str, header: list, row: list):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        with open(filepath, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            if write_header:
                w.writerow(header)
            w.writerow(row)

    @classmethod
    def write_bench_timings_row(cls, bench_dir: str, **cols) -> str:
        path = os.path.join(bench_dir, "bench_timings.tsv")
        row = [cols.get(k, "") for k in cls.BENCH_TIMINGS_HEADER]
        cls.append_row(path, cls.BENCH_TIMINGS_HEADER, row)
        return path

    @classmethod
    def upsert_pe_winners_row(cls, bench_dir: str, N: int,
                              residues_per_segment=None,
                              max_angle_hinge=None,
                              batch_size=None) -> str:
        """Merge a per-N winner into pe_winners.tsv without duplicating rows.

        PE1, PE2, PE3 each call this with a subset of the four columns populated.
        Existing values are preserved when the caller passes None for a column.
        """
        path = os.path.join(bench_dir, "pe_winners.tsv")
        rows: dict = {}
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f, delimiter="\t")
                for row in r:
                    rows[int(row["N"])] = {
                        "residues_per_segment": row.get("residues_per_segment", ""),
                        "max_angle_hinge": row.get("max_angle_hinge", ""),
                        "batch_size": row.get("batch_size", ""),
                    }
        existing = rows.get(int(N), {"residues_per_segment": "",
                                     "max_angle_hinge": "",
                                     "batch_size": ""})
        if residues_per_segment is not None:
            existing["residues_per_segment"] = residues_per_segment
        if max_angle_hinge is not None:
            existing["max_angle_hinge"] = max_angle_hinge
        if batch_size is not None:
            existing["batch_size"] = batch_size
        rows[int(N)] = existing

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(cls.PE_WINNERS_HEADER)
            for n_key in sorted(rows.keys()):
                rec = rows[n_key]
                w.writerow([n_key, rec["residues_per_segment"],
                            rec["max_angle_hinge"], rec["batch_size"]])
        return path

    @classmethod
    def read_pe_winners(cls, bench_dir: str) -> dict:
        """Return {N: {residues_per_segment, max_angle_hinge, batch_size}} from pe_winners.tsv."""
        path = os.path.join(bench_dir, "pe_winners.tsv")
        out: dict = {}
        if not os.path.exists(path):
            return out
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                N = int(row["N"])
                def _opt_int(s):
                    return int(s) if s not in (None, "", "None") else None
                def _opt_float(s):
                    return float(s) if s not in (None, "", "None") else None
                out[N] = {
                    "residues_per_segment": _opt_int(row.get("residues_per_segment", "")),
                    "max_angle_hinge": _opt_float(row.get("max_angle_hinge", "")),
                    "batch_size": _opt_int(row.get("batch_size", "")),
                }
        return out
