#!/usr/bin/env bash
# Short-schedule 12-cell matrix for time-bounded run.
# Uniform eq=20, prod=30 across all N keeps slope fits clean.
set -u
cd "D:/git" || exit 1
OUT="D:/git/rouse_model_python/bench_output"
mkdir -p "${OUT}/06_benchmarks"

EQ=20
PROD=30

cell_idx=0
total_cells=12
GLOBAL_T0=$(date +%s)

for N in 25 50 100; do
  for alg in multistep conventional; do
    for dev in gpu cpu; do
      cell_idx=$((cell_idx + 1))
      tag="B-N${N}-${alg}-${dev}"
      t_begin=$(date +%s)
      printf "[MATRIX %d/%d begin] %s  EQ=%d PROD=%d\n" "$cell_idx" "$total_cells" "$tag" "$EQ" "$PROD"
      if [ "$dev" = "gpu" ]; then parallel=true; else parallel=false; fi
      python -m rouse_model_python.src.apps.benchmark.main \
          --device "$dev" --is_parallel "$parallel" --force \
          --algorithm "$alg" \
          --output_dir "${OUT}" \
          --N "$N" --n_chains 300 --phi 0.10 \
          --eq_sweeps "$EQ" --prod_sweeps "$PROD" \
          --residues_per_segment 20 --max_angle_hinge 3.141593 \
          --warmup_sweeps 5 --repeats 1 --seed 42 \
          --cell_tag "${tag}" 2>&1 | grep -E "\[BENCH-(CFG|RUN|B3|CELL-END|B2)\]|ERROR|Traceback"
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

GLOBAL_T1=$(date +%s)
printf "[MATRIX COMPLETE] wrote %d cells in %ds\n" "$cell_idx" "$((GLOBAL_T1 - GLOBAL_T0))"
