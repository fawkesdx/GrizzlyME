# API

Public exports from `grizzly`:

| Symbol | Role |
|--------|------|
| `GrizzlyExperiment` | Drop-in wrapper around chinook `experiment` |
| `compute_all_Mk` | Batched matrix elements for valid peaks |
| `solve_H` | Batched H(k) diagonalization |
| `compute_fermi_distribution` | Fermi–Dirac weights |
| `compute_M_factor` | \|M·ε\|² factor (fixed or per-peak pol) |
| `compute_spectral_intensity` | Scatter-add Green’s function peaks |
| `gaussian_convolution_3d` | FFT Gaussian broadening |
| `get_device` | Device selection helper |
| `GrizzlyMEv2FeatureError` | Raised for spin / deferred features |

Module layout: `grizzly/experiment.py`, `engine.py`, `hamiltonian.py`, `spectral.py`, `radint_cache.py`, `ylm_torch.py`, `utils.py`, `future.py`.
