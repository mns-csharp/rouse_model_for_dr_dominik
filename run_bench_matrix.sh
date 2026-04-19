#!/usr/bin/env bash
# 12-cell Rouse MC benchmark matrix driver.
# Must run from D:/git/ so the module path rouse_model_python.* resolves.
# Each cell writes one row to bench_output/06_benchmarks/bench_timings.tsv.

set -u
cd "D:/git" || exit 1

OUT="D:/git/rouse_model_python/bench_output"
mkdir -p "${OUT}/06_benchmarks"

cell_idx=0
total_cells=12

for N in 25 50 100; do
  for alg in multistep conventional; do
    for dev in gpu cpu; do
      cell_idx=$((cell_idx + 1))
      tag="B-N${N}-${alg}-${dev}"
      t_begin=$(date +%s)
      printf "[MATRIX %d/%d begin] %s\n" "$cell_idx" "$total_cells" "$tag"
      if [ "$dev" = "gpu" ]; then parallel=true; else parallel=false; fi
      python -m rouse_model_python.src.apps.benchmark.main \
          --device "$dev" --is_parallel "$parallel" --force --timing_mode \
          --algorithm "$alg" \
          --output_dir "${OUT}" \
          --N "$N" --n_chains 300 --phi 0.10 \
          --residues_per_segment 20 --max_angle_hinge 3.141593 \
          --warmup_sweeps 10 --repeats 3 --seed 42 \
          --cell_tag "${tag}" 2>&1 | grep -E "\[BENCH-(CFG|RUN|B3|CELL-END)\]|ERROR|Traceback"
      rc=${PIPESTATUS[0]}
      t_end=$(date +%s)
      elapsed=$((t_end - t_begin))
      if [ "$rc" -ne 0 ]; then
        printf "[MATRIX %d/%d FAIL rc=%d] %s elapsed=%ds\n" "$cell_idx" "$total_cells" "$rc" "$tag" "$elapsed"
      else
        last_row=$(tail -n 1 "${OUT}/06_benchmarks/bench_timings.tsv" 2>/dev/null)
        printf "[MATRIX %d/%d done] %s elapsed=%ds\n  last_row=%s\n" "$cell_idx" "$total_cells" "$tag" "$elapsed" "$last_row"
      fi
    done
  done
done

printf "[MATRIX COMPLETE] wrote %d cells\n" "$cell_idx"
