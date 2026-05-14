"""cpu/single_core/conventional_mc/py_torch_fused — fully batched per-sweep MC.

Differences vs `c1c_pt_cpu_single_core_conventional_mc_py_torch`:
  - Per sweep proposes ALL N*K segments in one batched call (no Python loop).
  - Computes delta_E for all N*K proposals in one fused torch.cdist (no `for b in range(B)`).
  - Applies parallel-Gibbs Metropolis accept/reject vectorized across all segments.
  - Persistent torch tensors for state; no per-iteration torch.from_numpy.

Trade-off: accept decisions use a frozen reference state from the start of the
sweep, so this is a parallel-Gibbs sampler rather than strict sequential
Metropolis. For Rouse hinge moves with anchored endpoints, segments do not
share moved beads, so within-chain detail balance is preserved; cross-chain
moves are computed against stale reference but at most one move applies per
chain region, so collisions cannot stack. Acceptable bias for high-throughput
runs; validate against the sequential app on Kuriata observables.

SURPASS-alpha CG: Calpha + SG bead per residue, l0 = sigma.
"""
