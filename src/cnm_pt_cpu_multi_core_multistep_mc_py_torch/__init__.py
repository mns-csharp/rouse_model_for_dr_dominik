"""cpu/single_core/multistep_mc/py_torch — torch CPU tensors, single-thread.

Same physics as the numba sibling; engine differs:
  - Hot ops written as torch CPU tensor ops (cdist, bmm, einsum).
  - torch.set_num_threads(1) enforces serial execution.
  - Sequential accept loop runs on host with the per-batch dE/correction
    read back to numpy.

SURPASS-alpha CG: Calpha + SG bead per residue, l0 = sigma.
"""
