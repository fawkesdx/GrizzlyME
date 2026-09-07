# Vision & roadmap

## Core objective

GPU-accelerated ARPES **matrix-element** simulation as a high-performance companion to the CPU-bound [chinook](https://github.com/rpday/chinook) package — modular, installable, and aimed at the broader ARPES community (PyPI/Conda when ready).

## Design targets

| Target | Status (v0.2) |
|--------|----------------|
| Batched ME + spectral on CPU / CUDA | **Done** (spinless) |
| Drop-in `GrizzlyExperiment` API | **Done** |
| Large-VRAM NVIDIA GPUs (e.g. Tesla V100 class) | **Supported** (single-GPU CUDA) |
| Multi-GPU / very large slabs | Partial (chunked batches); full multi-GPU not yet |
| Public install (GitHub) | **Done** |
| PyPI / Conda | **Not yet** |
| Spin / SARPES | **Deferred** (clear error in v0.2) |
| Faster first-call radials | **Done** in v0.2.0 via `prepare_me_shell` (torch `slater` / `hydrogenic`); stock `GrizzlyExperiment.datacube()` wiring pending |

## Honest speedups

Early planning hoped for ~100–1000× on full maps. Measured reality for v0.2:

- **Matrix elements alone:** large gains (often tens of × on bigger cubes; higher on CUDA)
- **TensorSpec Full-ME / `prepare_me_shell`:** torch removes the slow first radial-integral build for `slater` / `hydrogenic`
- **Stock `GrizzlyExperiment.datacube()`:** cold radial setup still uses chinook until wired to the shell path; repeat calls benefit from the cache

Repeat runs on the same model benefit strongly from the radint cache.

## Next (community-facing)

1. PyPI (and optionally Conda) release of `grizzlyme`
2. Broader material examples in docs/tests
3. Spin / SARPES (v2)
4. Torch radial integrals for the `grid` / `exec` `rad_type` paths
