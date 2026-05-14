"""cpu/single_core/conventional_mc/py_torch — torch CPU tensors, single-thread.

Segmented sequential Metropolis. Cell list rebuilt every proposal by
re-creating torch tensor views over the mutated numpy state.

SURPASS-alpha CG: Calpha + SG bead per residue, l0 = sigma.
"""
