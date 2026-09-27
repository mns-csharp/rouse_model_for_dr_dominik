# App codes

27 apps under `src/`. (Two PyTorch-multistep apps `g1m_pt` and `gnm_pt`
were removed on 2026-05-14 after they consistently timed out at
N=100, K=20. Two optimized CUDA-C multistep apps `g1m_ccx` and `gnm_ccx`
were added on 2026-05-14 — new versions of `g1m_cc` / `gnm_cc`, not edits
to them; the original 25 apps are frozen.)

Each app under `src/` has a mnemonic code prefix encoding its nature:

```
Code: <hw><thread><algo>_<backend>
  hw      : c = CPU,  g = GPU
  thread  : 1 = single (single_core / single_thread),  n = multi (multi_core / multi_thread)
  algo    : c = conventional Metropolis MC,  m = Migacz multistep MC
  backend : nb    = numba
            pt    = py_torch
            cc    = cuda_c
            ptc   = py_torch_compiled            (CPU exploratory variant)
            ptf   = py_torch_fused               (CPU exploratory variant)
            ptg   = py_torch_gpu_fused           (GPU-resident, fully batched, torch.compile-wrapped)
            ptgcl = py_torch_gpu_fused_cell_list (ptg + spatial cell-list neighbour lookup; large-K)
            cccl  = cuda_c_cell_list             (hand-written CUDA-C kernel with 27-cell scan)
            ccx   = cuda_c_streamed              (optimized CUDA-C multistep: whole-grid streamed ΔE,
                                                  sparse correction kernel, on-GPU causal accept)
```

The full directory name is `<code>_<descriptive_name>` so both ways of reading it work. Import paths use the new directory name verbatim, e.g.:

```
python -m g1c_ptg_gpu_single_thread_conventional_mc_py_torch_gpu_fused.main --N 100 ...
```

## Code → app → purpose

| Code      | Directory                                                                   | Purpose |
|-----------|-----------------------------------------------------------------------------|---------|
| `c1c_nb`  | c1c_nb_cpu_single_core_conventional_mc_numba                                | CPU Numba baseline (single-threaded). The reference implementation; fastest CPU choice for small/mid sizes. |
| `c1c_pt`  | c1c_pt_cpu_single_core_conventional_mc_py_torch                             | CPU PyTorch port of the baseline; loses to Numba on CPU. |
| `c1c_ptc` | c1c_ptc_cpu_single_core_conventional_mc_py_torch_compiled                   | Exploratory: same per-segment loop, wrapped with `torch.compile`. Result was a wash with `c1c_pt`. |
| `c1c_ptf` | c1c_ptf_cpu_single_core_conventional_mc_py_torch_fused                      | Exploratory: round-based fused dispatch on CPU. ~1.6× faster than `c1c_pt` but still loses to `c1c_nb`. |
| `c1m_nb`  | c1m_nb_cpu_single_core_multistep_mc_numba                                   | CPU Numba multistep (rank-1 EMM correction). |
| `c1m_pt`  | c1m_pt_cpu_single_core_multistep_mc_py_torch                                | CPU PyTorch multistep port. |
| `cnc_nb`  | cnc_nb_cpu_multi_core_conventional_mc_numba                                 | CPU Numba parallel (prange). |
| `cnc_pt`  | cnc_pt_cpu_multi_core_conventional_mc_py_torch                              | CPU PyTorch parallel (OpenMP via torch). |
| `cnm_nb`  | cnm_nb_cpu_multi_core_multistep_mc_numba                                    | CPU Numba parallel multistep (prange + rank-1 EMM correction). |
| `cnm_pt`  | cnm_pt_cpu_multi_core_multistep_mc_py_torch                                 | CPU PyTorch parallel multistep. |
| `g1c_cc`  | g1c_cc_gpu_single_thread_conventional_mc_cuda_c                             | Hand-written CUDA-C, single-stream conventional. Cupy RawKernel via NVRTC. Single fully-fused mega-kernel per sweep. |
| `g1c_pt`  | g1c_pt_gpu_single_thread_conventional_mc_py_torch                           | Original GPU PyTorch (per-segment Python loop antipattern; the "PyTorch is 60× slower" baseline). |
| `g1c_ptg` | g1c_ptg_gpu_single_thread_conventional_mc_py_torch_gpu_fused                | **Redesigned PyTorch on GPU.** Round-based fused dispatch, `torch.compile`. Portable alternative to the hand-written CUDA-C kernel. |
| `g1m_cc`  | g1m_cc_gpu_single_thread_multistep_mc_cuda_c                                | Hand-written CUDA-C Migacz multistep: round-permutation batching, CUDA-C propose kernel, batched ΔE, rank-1 EMM correction matrix, host-side causal accept loop. |
| `g1m_ptg` | g1m_ptg_gpu_single_thread_multistep_mc_py_torch_gpu_fused                   | **Redesigned PyTorch multistep on GPU**, fully vectorised correction matrix. |
| `gnc_cc`  | gnc_cc_gpu_multi_thread_conventional_mc_cuda_c                              | CUDA-C with multiple CUDA streams. |
| `gnc_pt`  | gnc_pt_gpu_multi_thread_conventional_mc_py_torch                            | Original GPU PyTorch with streams. |
| `gnc_ptg` | gnc_ptg_gpu_multi_thread_conventional_mc_py_torch_gpu_fused                 | **Redesigned PyTorch + CUDA streams** on top of fused dispatch. |
| `gnm_cc`  | gnm_cc_gpu_multi_thread_multistep_mc_cuda_c                                 | Hand-written CUDA-C Migacz multistep with multi-stream propose+ΔE dispatch; rank-1 correction computed on the default stream after stream rejoin. |
| `gnm_ptg` | gnm_ptg_gpu_multi_thread_multistep_mc_py_torch_gpu_fused                    | **Redesigned PyTorch streams + multistep**. |
| `g1c_ptgcl` | g1c_ptgcl_gpu_single_thread_conventional_mc_py_torch_gpu_fused_cell_list | **Redesigned PyTorch + cell-list neighbour lookup.** 27-cell spatial neighbour scan on top of fused dispatch; designed for large chain counts where all-pairs cost dominates. |
| `g1m_ptgcl` | g1m_ptgcl_gpu_single_thread_multistep_mc_py_torch_gpu_fused_cell_list | Multistep MC variant of `g1c_ptgcl`. |
| `gnc_ptgcl` | gnc_ptgcl_gpu_multi_thread_conventional_mc_py_torch_gpu_fused_cell_list | Stream variant of `g1c_ptgcl`. |
| `gnm_ptgcl` | gnm_ptgcl_gpu_multi_thread_multistep_mc_py_torch_gpu_fused_cell_list | Streams + multistep + cell-list. |
| `g1c_cccl` | g1c_cccl_gpu_single_thread_conventional_mc_cuda_c_cell_list | **Hand-written CUDA-C cell-list.** 27-cell scan in a hand-written kernel; the cell-list counterpart to `g1c_cc`, designed for large chain counts. |
| `g1m_ccx` | g1m_ccx_gpu_single_thread_multistep_mc_cuda_c_streamed | **Optimized CUDA-C multistep** (new version of `g1m_cc`). Whole-grid streamed ΔE kernel (`grid=(B,1,1)`), sparse correction kernel with an exact bounding-sphere prefilter (`grid=(B,B,1)`), on-GPU causal accept + scatter — zero per-batch GPU↔host round-trips. Beats `g1c_cc`/`gnc_cc` for K ≥ 128. |
| `gnm_ccx` | gnm_ccx_gpu_multi_thread_multistep_mc_cuda_c_streamed | Multi-thread sibling of `g1m_ccx` (new version of `gnm_cc`). Same streamed CUDA-C kernels; the per-batch CUDA-stream split was dropped because the whole-grid kernels already saturate the GPU. `--streams` accepted for CLI parity but inert. |

## Benchmark

The authoritative current benchmark is the run delivered in
`D:\git\RouseModel\05_benchmarks\hinge_opt_series\rouse_python_benchmark_hinge_opt_2026-05-14_003001\` — 25 apps ×
N ∈ {25, 50, 100} × K = 20, eq = prod = 100, seed = 42, phi = 0.01. See its
`README.md`, `benchmark.md`, `results.tsv`, and `tables.md` for the full
results.

Headline ranking — throughput at N = 100, K = 20 (top 7 of 25; baseline
`c1c_nb` is rank 13):

| Rank | Code       | Throughput (sw/s) | Speedup × vs `c1c_nb` |
|------|------------|-------------------|------------------------|
| 1    | `gnc_cc`   | 75.67             | 49.87                  |
| 2    | `g1c_cc`   | 71.96             | 47.42                  |
| 3    | `g1c_cccl` | 11.88             |  7.83                  |
| 4    | `g1m_cc`   |  6.86             |  4.52                  |
| 5    | `g1c_ptg`  |  5.62             |  3.70                  |
| 6    | `g1m_ptg`  |  4.19             |  2.76                  |
| 7    | `gnm_cc`   |  3.46             |  2.28                  |
| 13   | `c1c_nb`   |  1.52             |  1.00 (baseline)       |

The two conventional CUDA-C apps lead. The multistep apps trail conventional
at K = 20 — consistent with the Migacz paper, which identifies small batch
sizes as the inefficient regime; see the deliverable's README §7.1 / §7.5.
