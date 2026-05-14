"""gpu/single_thread/multistep_mc/cuda_c — PyCuPy RawKernel-augmented CUDA (uses torch.cuda for tensor allocation; selected hot loops via cupy.RawKernel) backend.

Self-contained. Single CUDA stream. Segmented Migacz multistep MC with
dense BxB rank-1 correction. Calpha + SG SURPASS-alpha CG (l0 = sigma).
All hot operations run as CuPy RawKernel-augmented CUDA (uses torch.cuda for tensor allocation; selected hot loops via cupy.RawKernel) tensor ops; the only host-side work
is the round-permutation builder and (each batch) the rank-1 accept loop
via fused_accept on numpy.
"""
