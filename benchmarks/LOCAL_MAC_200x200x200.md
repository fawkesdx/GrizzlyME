# Local Mac Studio — 200×200×200 Chinook ARPES (baseline)

**Do not delete.** Paper / SI comparison vs cuda-host GrizzlyME 200³.
**Status: COMPLETED** 2026-08-30.

## Job identity

| Field | Value |
|-------|--------|
| Host | Mac Studio (Apple Silicon) |
| Target | Local in-process Chinook path (not remote runner) |
| TB | production Wannier class (~107 MB; ~6.7M hoppings) |
| Physics | hv=84, workf=4.5, V0=12, T=10 K, p-pol, hkl=(-2,0,1), incidence 55°, slit 0° |
| Grid | **200 × 200 × 200** confirmed (`intensity` shape) |
| Diag | **175** segments; excluded **8,160,000** exact-zero eigenvalues |

## Final timing (recorded)

| Metric | Value |
|--------|-------|
| **wall_s (GUI start → cube mtime)** | **37187 s (~10.33 h)** |
| wall_s (GUI start → first 100% bar) | ~35211 s (~9.78 h) |
| full-cube wall line | not printed (GUI Chinook path) |
| completed | **yes** |
| cube shape | **(200, 200, 200)** float64 `intensity` |
| cube bytes | **60,844,663** (~58.0 MB) |
| peak RAM (observed) | ~211 GB footprint |

### Paper one-liner

Local Mac Studio Chinook (Apple Silicon, GUI in-process) produced a validated **200³** ARPES intensity cube in **~10.3 h** wall time, with peak memory on the order of **~200 GB**.

## cuda-host compare

See `CUDA_HOST_200x200x200.md` for full provenance.

| | Mac local Chinook | cuda-host Grizzly (1× V100) | cuda-host Grizzly (2× V100 est.) |
|--|-------------------|-----------------------------|----------------------------------|
| wall_s | **37187 (~10.33 h)** | **762 (~12.7 min)** | **~1970 (~32.8 min)** |
| speedup vs Mac | 1× | **~49×** | **~19×** |
| device | Apple Silicon CPU | V100, θ-chunk=20 | 2× V100, θ-chunk=18 |

Mid-grid cuda-host ladder (not full 200³): `PAPER_SCALE_LADDER.md`.
