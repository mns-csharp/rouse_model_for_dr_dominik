"""cpu/single_core/conventional_mc/py_torch_compiled — same algorithm as
`c1c_pt_cpu_single_core_conventional_mc_py_torch`, but with `torch.compile` applied
to `batch_delta_e_torch` and `propose_batch_torch`.

The per-segment Python for-loop in `mc.perform_sweep` is preserved. The
expectation: each iteration's tensor work is compiled into a single
inductor C++ kernel, so per-iteration dispatch overhead drops.

Compile backend selection via ROUSE_TORCH_COMPILE_MODE env var
(inductor | aot_eager | script | eager; default inductor). On Windows
inductor requires `cl.exe` on PATH (activate vcvarsall.bat).

SURPASS-alpha CG: Calpha + SG bead per residue, l0 = sigma.
"""
