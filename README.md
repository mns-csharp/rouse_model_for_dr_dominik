# Rouse MC simulator: benchmark + T1-T7 validation

PyTorch Rouse-model simulator with two entry-point apps:

- **Benchmark** (`src/apps/benchmark/`): the 12-cell performance
  matrix that compares conventional single-trial Metropolis MC against
  the Migacz rank-1 multistep MC on both CPU and GPU.
- **Validation** (`src/apps/validation/`): the T1-T7 Rouse-property
  validator. For each (N, phi) cell it fits the static and dynamic
  Rouse scalings (end-to-end distance, radius of gyration, COM
  diffusion, g1 short-time, Rouse time tau_R, diffusion D) and emits
  PASS/FAIL verdicts per the T1..T7 thresholds.

## Important: the top-level directory must be named `rouse_model_python`

Every Python import in this tree starts with `rouse_model_python`
(for example `from rouse_model_python.src.apps.benchmark.orchestrator
import BenchmarkOrchestrator`). For those imports to resolve, the
directory containing this code must be named `rouse_model_python`.
If it currently has a different name, rename it:

```bash
mv <current-folder-name> rouse_model_python
```

Then run all commands from the parent directory so Python locates
the package.

## Hardware prerequisites

- NVIDIA GPU with CUDA 12.x for the benchmark matrix (RTX 4070 or
  better recommended). The benchmark has CPU-only cells too, but the
  headline Migacz speedup only shows up on GPU.
- The validation app runs on CPU or GPU.

## Install

```bash
pip install -r requirements.txt
```

On Windows, also install the matching Triton build to keep
`torch.compile` out of eager-mode fallback:

```bash
pip install triton-windows==3.2.0
```

Triton must match the torch version: torch 2.6 pairs with triton
3.2. Without it, the GPU multistep kernels lose about a factor of
four.

## Sanity checks after install

```bash
cd <parent-of-rouse_model_python>
python -m rouse_model_python.src.apps.benchmark.orchestrator --help
python -m rouse_model_python.src.apps.validation.main --help
```

If either import fails, check that the package directory is named
exactly `rouse_model_python`.

## Running the benchmark

The full protocol (PE1/PE2/PE3 pre-experiments, 12-cell main matrix,
acceptance checks, plots) is specified in `context_rouse_benchmark.txt`
and driven by `src/apps/benchmark/orchestrator.py`. A typical
invocation from the parent directory:

```bash
python -m rouse_model_python.src.apps.benchmark.orchestrator \
    --output_dir bench_output --phi 0.035 --n_chains 50 \
    --warmup_sweeps 5 --repeats 3 --seed 42 --b_timing_mode \
    --only r7,b1,b2,pe1,pe2,pe3,b_matrix,analyze
```

A fast end-to-end run with the reduced sweep schedule takes about
1.5 h on an RTX 4070 for the single-replica matrix, about 4 h for
the R=3 companion run. The four shell scripts at the project root
(`run_bench_matrix.sh`, `run_bench_matrix_short.sh`,
`run_migacz_torch_bench.sh`, `run_migacz_torch_bench_large.sh`) are
convenience wrappers around the Python orchestrator; the
orchestrator itself is the authoritative runner.

## Running the validation

The Rouse-property acceptance checks are specified in
`context_rouse_verification.txt` and driven by
`src/apps/validation/main.py`. Minimal invocation:

```bash
python -m rouse_model_python.src.apps.validation.main \
    --device cpu --is_parallel false --force \
    --output_dir validation_output
```

The app walks the N x phi matrix from `src/configs/simulation.toml`,
runs the simulation for each cell, fits the seven Rouse scalings,
and writes per-cell TSVs plus `00_t1_t7_verdicts.json` under the
chosen `--output_dir`.

## Directory map

```
rouse_model_python/                     (required folder name)
  __init__.py
  README.md                              (this file)
  requirements.txt
  context_rouse_benchmark.txt            (benchmark protocol spec)
  context_rouse_verification.txt         (T1-T7 validation protocol spec)
  run_bench_matrix.sh                    (optional benchmark wrapper)
  run_bench_matrix_short.sh
  run_migacz_torch_bench.sh
  run_migacz_torch_bench_large.sh
  src/
    __init__.py
    apps/
      __init__.py
      benchmark/                         (orchestrator, BenchmarkApp, main)
      validation/                        (T1T7ValidatorApp, main)
    configs/                             (benchmark.toml, physics.toml,
                                          simulation.toml, execution.toml)
    libs/                                (algorithms, analysis, chain,
                                          config, energy, execution, io,
                                          mc_moves, number_space,
                                          observables, paths, simulation)
```

## Citation

The Migacz multistep algorithm is described in:

Migacz, S.; Dutka, K.; Gumienny, P.; Marchwiany, M.; Gront, D.;
Rudnicki, W. R. "Parallel Implementation of a Sequential Markov
Chain in Monte Carlo Simulations of Physical Systems with Pairwise
Interactions." *J. Chem. Theory Comput.* 2019, **15**, 2797-2806.
