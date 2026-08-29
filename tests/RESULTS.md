# GrizzlyME test results

- when: 2026-08-27T20:47:00Z
- gate: Phase 5 close (Task 5c) — `tests/test_spectral.py` + `tests/test_full_pipeline.py` + `tests/test_hamiltonian.py`
- result: **13 passed**, 0 failed
- per file:
  - `tests/test_spectral.py`: **5 passed**
  - `tests/test_full_pipeline.py`: **3 passed**
  - `tests/test_hamiltonian.py`: **5 passed**
- failed: (none)
- benchmarks: see `benchmarks/RESULTS.md` (Phase 6 Tasks 6a–6c; 6c skipped no CUDA)
- publication stub: `docs/PUBLICATION_NOTES.md`

## Phase 6 close (Task 6d, 2026-08-28)

- gate: `pytest tests/test_*.py -q` → **21 passed**
- Phase 6 benchmarks written; see `benchmarks/RESULTS.md`

## Phase 7 close (2026-08-28)

- `README.md`, `pyproject.toml`, `PLAN.md` §9 updated
- gate: **23 passed**
- publication stub: `docs/PUBLICATION_NOTES.md`
- **Phases 0–7 complete**

## Finish line (2026-08-28, post Phase 7)

- Second TB model: Si-like 8-orbital s/p (`tests/test_second_model.py`) — Mk + spectral parity
- Bugfix: angle-mode `gen_all_pol` in `compute_M_factor` / `build_raw_I_from_experiment`
- v1 scope table in `README.md`
- Profile: `benchmarks/profile_datacube.py` — setup (radint/basis/pks) dominates wall time
- Production CUDA host: ME **63–75×** at 30–40³; full pipeline **1.39–1.47×**
- gate: **28 passed**
- `plot_arpes_comparison.py` → `GrizzlyExperiment` API; figure in `docs/figures/`

## Phase 4 — spectral (`test_spectral.py`, 5/5)

- `compute_fermi_distribution`: chinook parity (kb/q, overflow 709)
- `compute_M_factor`: spinless momentum projection
- `compute_spectral_intensity`: raw I vs chinook ~1e-16 rel L2
- `gaussian_convolution_3d`: FFT circular vs scipy reflect (~1e-3 on tiny cubes; Ig not hard-gated)
- end-to-end spectral assembly on graphene TB

## Phase 3 — hamiltonian (`test_hamiltonian.py`, 5/5)

- `extract_hopping_data` + `build_and_diagonalize` vs NumPy eigh (< 1e-10)
- `solve_H` vs chinook `TB_model.solve_H` on graphene
- eigenvector gauge consistency
- chunked k-mesh diagonalization
- Hermitian upper-triangle (UPLO='U') parity

## Phase 5 — experiment (`test_full_pipeline.py`, 3/3)

- Task 5a: `GrizzlyExperiment` datacube + `ensure_Mk()` on graphene; Mk shape `(N,2,3)`, finite
- Task 5b: `spectral()` via `build_raw_I_from_experiment` + `gaussian_convolution_3d`
  - `test_spectral_raw_I_vs_chinook` (shared Mk): raw I rel L2 **2.13e-16** (< 1e-4); Ig rel L2 **5.66e-3** (< 0.01)
- Task 5c: `grizzly/__init__.py` public exports + `test_cpu_device`
  - `GrizzlyExperiment(..., device='cpu')`: datacube + ensure_Mk on CPU
  - package import: `from grizzly import GrizzlyExperiment`

## Package exports (`grizzly/__init__.py`)

- `GrizzlyExperiment`, `compute_all_Mk`
- `solve_H`, `extract_hopping_data`, `build_and_diagonalize`
- `compute_fermi_distribution`, `compute_M_factor`, `compute_spectral_intensity`, `gaussian_convolution_3d`
- `get_device`
