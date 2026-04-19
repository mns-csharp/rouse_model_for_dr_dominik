"""Entry point for the benchmark app.

Usage:
    python -m rouse_model_python.src.apps.benchmark.main \
        --device gpu --is_parallel false --force \
        --algorithm multistep --N 100 --n_chains 50 --phi 0.035 \
        --residues_per_segment 20 --max_angle_hinge 3.141593 \
        --warmup_sweeps 50 --cell_tag B-N100-multistep-gpu
"""

import sys

from rouse_model_python.src.apps.benchmark.benchmark_app import BenchmarkApp


def main(argv=None) -> int:
    return BenchmarkApp(argv=argv).run()


if __name__ == "__main__":
    sys.exit(main())
