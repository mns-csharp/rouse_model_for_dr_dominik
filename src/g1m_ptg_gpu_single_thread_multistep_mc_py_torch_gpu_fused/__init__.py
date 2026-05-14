"""gpu/single_thread/multistep_mc/py_torch — PyTorch CUDA backend.

Self-contained. Single CUDA stream. Segmented Migacz multistep MC with
dense BxB rank-1 correction. Calpha + SG SURPASS-alpha CG (l0 = sigma).
All hot operations run as torch CUDA tensor ops; the only host-side work
is the round-permutation builder and (each batch) the rank-1 accept loop
via fused_accept on numpy.
"""
