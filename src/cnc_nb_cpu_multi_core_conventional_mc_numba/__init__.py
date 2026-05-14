"""cpu/multi_core/conventional_mc — sequential per-proposal Metropolis MC, multicore CPU.

Self-contained. SURPASS-alpha CG: Calpha + SG bead per residue.

Conventional MC's accept/reject loop is sequential by definition (each
proposal's outcome must be known before the next is generated), so the
multicore parallelism here is *inside* the energy computation:
  - Parallel cell-list build (Numba prange over beads).
  - Parallel union-of-cells delta-E evaluation (Numba prange over the
    visited-cell list, with per-thread accumulators reduced at the end).
"""
