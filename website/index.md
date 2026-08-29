# GrizzlyME

Faster matrix elements and spectral assembly for [chinook](https://github.com/rpday/chinook) ARPES simulations, implemented in PyTorch.

!!! info "v0.1 alpha"
    Supported today: **spinless** tight-binding models. Spin / SARPES are planned for a later release and raise a clear error if requested.

## What it does

Chinook remains responsible for building the TB model and (on first use) radial integrals. GrizzlyME accelerates:

- Batched photoemission matrix elements (`compute_all_Mk`)
- Hamiltonian diagonalization over the k-mesh (`solve_H`)
- Spectral intensity assembly and Gaussian broadening

Typical usage is a one-line swap: `GrizzlyExperiment` instead of `chinook.ARPES_lib.experiment`.

## Name

**Grizzly** honors Grizzly Peak near Berkeley Lab / MAESTRO (institutionally independent package). **ME** = matrix element.  
**G.R.I.Z.Z.L.Y.** = GPU Rendered Inner-products for Zone-by-Zone Lattice Yields.  
Details: [Name & branding](naming.md).

## Links

- [Install](install.md)
- [Usage](usage.md)
- [Scope & limits](limits.md)
- [Name & branding](naming.md)
- [Vision & roadmap](roadmap.md)
- [Source on GitHub](https://github.com/fawkesdx/GrizzlyME)
