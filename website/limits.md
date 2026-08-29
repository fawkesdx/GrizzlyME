# Scope & limits

## Supported (v0.1)

| Feature | Notes |
|---------|--------|
| Spinless TB | `spin={'bool': False}` |
| Momentum cubes | `X`, `Y`, `E` |
| Angle cubes | `Tx`, `Ty`, `E` (uses chinook `gen_all_pol`) |
| Multi-orbital bases | Validated on graphene π and an 8-orbital s/p test model |
| CPU and CUDA | `device='cpu'` / `'cuda'` / `'auto'` |

## Not in v0.1

| Feature | Behavior |
|---------|----------|
| Spin / SARPES | Raises `GrizzlyMEv2FeatureError` |
| MPS float64 | Unsupported — use CPU |
| Chinook interactive plot GUI | Unchanged; use your own plotting |
| Bit-identical `Ig` vs scipy `gaussian_filter` | FFT circular vs reflect; small relative difference possible |

## Correctness

Automated tests compare GrizzlyME to chinook on:

- Matrix elements (tight absolute tolerances)
- Raw spectral intensity with shared eigenpairs
- Hamiltonian eigenvalues / eigenvectors

See `tests/` in the repository. Run `pytest tests/ -q` after install.

## What “alpha” means

The accelerated path is exercised and parity-tested for the supported scope. Unusual chinook options (custom `rad_type`, slab truncation, spin) may be untested or rejected. Report issues on GitHub.
