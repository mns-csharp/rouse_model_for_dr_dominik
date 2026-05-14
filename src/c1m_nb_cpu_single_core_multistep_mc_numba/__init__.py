"""cpu/single_core/multistep_mc — Migacz multistep MC, single-core CPU.

Self-contained ground-truth reference. SURPASS-alpha CG: Calpha + SG bead
per residue, rigid Calpha-Calpha bonds, athermal excluded volume.

Implements:
  - Hinge + tail-N + tail-C moves (rigid Calpha+SG together)
  - Chain segmentation
  - Migacz multistep with full B x B rank-1 correction matrix
  - NumberSpace MIC/PBC
  - No replicas, no batching across replicas
  - Numba @njit serial kernels (no parallel=True)
"""
