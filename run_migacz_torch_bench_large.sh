#!/usr/bin/env bash
# PyTorch Migacz multistep MC perf sweep on GPU — LARGE-N matrix.
# Matrix: N in {250, 500, 1000, 2000} x n_chains = 100 at phi=0.10.
# Uses --algorithm multistep --accept_on_gpu true (PyTorch path).
#
# N>=1000 would OOM a 12 GiB card under the default delta_e allocation,
# so we cap --delta_e_mem_budget_gib 10 to force batch-axis tiling.
#
# Per-cell timings append to gpu_lab_out/gpu_lab_runs.tsv; live GPU
# util/mem/power streams to stdout.
#
# Must run from D:/git/ so the module path rouse_model_python.* resolves.

set -u
cd "D:/git" || exit 1

OUT_DIR="rouse_model_python/gpu_lab_out"
mkdir -p "${OUT_DIR}"

N_LIST=(250 500 1000 2000)
NC=100

EQ=20
PROD=30
WARMUP=5
DELTA_E_BUDGET=10

cell_idx=0
total_cells=${#N_LIST[@]}
GLOBAL_T0=$(date +%s)

for N in "${N_LIST[@]}"; do
  cell_idx=$((cell_idx + 1))
  tag="migacz-torch-large-N${N}-nc${NC}"
  t_begin=$(date +%s)
  printf "\n[MATRIX %d/%d begin] %s  EQ=%d PROD=%d WARMUP=%d budget=%d GiB\n" \
         "$cell_idx" "$total_cells" "$tag" "$EQ" "$PROD" "$WARMUP" "$DELTA_E_BUDGET"
  python -m rouse_model_python.src.apps.gpu_lab.main \
      --device gpu --is_parallel true --force \
      --algorithm multistep \
      --accept_on_gpu true \
      --delta_e_mem_budget_gib "$DELTA_E_BUDGET" \
      --N "$N" --n_chains "$NC" --phi 0.10 \
      --eq_sweeps "$EQ" --prod_sweeps "$PROD" --warmup_sweeps "$WARMUP" \
      --seed 42 \
      --output_dir "${OUT_DIR}" \
      --tag "${tag}" \
      --live_print true
  rc=$?
  t_end=$(date +%s)
  elapsed=$((t_end - t_begin))
  if [ "$rc" -ne 0 ]; then
    printf "[MATRIX %d/%d FAIL rc=%d] %s elapsed=%ds\n" \
           "$cell_idx" "$total_cells" "$rc" "$tag" "$elapsed"
  else
    last_row=$(tail -n 1 "${OUT_DIR}/gpu_lab_runs.tsv" 2>/dev/null)
    printf "[MATRIX %d/%d done] %s elapsed=%ds\n  last_row=%s\n" \
           "$cell_idx" "$total_cells" "$tag" "$elapsed" "$last_row"
  fi
done

GLOBAL_T1=$(date +%s)
printf "\n[MATRIX COMPLETE] wrote %d cells in %ds to %s/gpu_lab_runs.tsv\n" \
       "$cell_idx" "$((GLOBAL_T1 - GLOBAL_T0))" "$OUT_DIR"
