"""cpu/multi_core/multistep_mc — Migacz multistep MC, true multicore CPU.

Self-contained, optimized for performance. SURPASS-alpha CG: Calpha + SG
bead per residue, rigid Calpha-Calpha bonds, athermal excluded volume.

Implements:
  - Hinge + tail-N + tail-C moves (rigid Calpha+SG together)
  - Chain segmentation with round-based ordering
  - Migacz multistep with sparse rank-1 correction (only overlapping pairs)
  - Numba @njit(parallel=True) over batch dim B for proposal / delta-E /
    correction kernels; sequential accept (rank-1 update is causal).
  - Pre-drawn host RNG so results are deterministic regardless of thread count.
  - No replicas; one simulation per process.
"""
