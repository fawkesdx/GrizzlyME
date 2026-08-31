# CUDA cluster — 200×200×200 GrizzlyME CUDA ARPES (paper record)

**Do not delete.** Production-scale endpoint for GrizzlyME vs local Mac Chinook baseline.
**Status:** Run A **logged**; Run B **completed** (wall estimated).

Public summary: `RESULTS.md` · Mac baseline: `LOCAL_MAC_200x200x200.md` · Mid-grid ladder: `PAPER_SCALE_LADDER.md`.

---

## Job identity (all runs)

| Field | Value |
|-------|--------|
| Host | cuda-host (1× or 2× NVIDIA V100 32GB) |
| TB | Production Wannier `tb_data.npz` (~107 MB; ~6.7M hoppings, 408 bands) |
| Physics | hv=84 eV, workf=4.5, V0=12, T=10 K, p-pol, hkl=(-2,0,1) |
| Grid | **200 × 200 × 200** (θ × φ × E), θ,φ ∈ [−15°, 15°], E ∈ [−1.0, 0.1] eV |
| Engine | GrizzlyME CUDA, layout **full**, hybrid path (Grizzly ME + Chinook spectral) |
| Runner | remote ARPES runner (TensorSpec GUI / SSH dispatch) |
| Output | `chinook_arpes_cube.npz` |

---

## Run A — single-GPU, logged wall (canonical for paper table)

| Metric | Value |
|--------|-------|
| **full-cube wall** | **762 s (~12.7 min)** — from log line `full-cube wall: 762.96s` |
| Finished | 2026-08-30 (cube mtime) |
| GPUs | **1** (`CUDA_VISIBLE_DEVICES=0`) |
| θ-chunk | **20** (10 blocks × ~76 s each) |
| OOM | no |
| Cube size | ~28 MB |

---

## Run B — dual-GPU multi-chunk (wall estimated)

| Metric | Value |
|--------|-------|
| **wall (estimated)** | **~1970 s (~32.8 min)** |
| Estimate method | Process elapsed + cube mtime (log overwritten by later rerun) |
| GPUs | **2** — log: `Using GPU ids: [0, 1]` |
| Layout | `FULL layout MULTI-GPU (θ-chunk=18, gpus=[0, 1]): 200 x 200 x 200, 12 blocks` |
| θ-chunk | **18** (auto VRAM plan) |
| OOM | no |
| Cube size | **29,163,587 B (~27.8 MB)** |
| Notes | Multi-GPU path; slower than Run A in this snapshot — scheduling overhead TBD |

---

## Mac vs cuda-host (paper compare)

| | Mac Studio Chinook | cuda-host Grizzly (Run A) | cuda-host Grizzly (Run B est.) |
|--|-------------------|---------------------------|--------------------------------|
| wall_s | **37,187 (~10.33 h)** | **762 (~12.7 min)** | **~1,970 (~32.8 min)** |
| speedup vs Mac | 1× | **~49×** | **~19×** |
| device | Apple Silicon CPU | 1× V100 | 2× V100 |
| path | GUI in-process Chinook | hybrid Grizzly ME + Chinook spectral | same + multi-GPU θ blocks |
| cube bytes | ~60.8 MB | ~28 MB | ~29.2 MB |

**Paper one-liner (draft):** On a production Wannier tight-binding model (~6.7M hoppings), a **200³** ARPES intensity cube that required **~10.3 h** on a Mac Studio running in-process Chinook completed in **12.7 min** on a single Tesla V100 using GrizzlyME CUDA full layout with θ-chunking — a **~49×** wall-clock reduction.

---

## Reproduce (generic)

```bash
# After staging tb_data.npz + physics on a CUDA node:
python -u chinook_remote_runner.py \
  --tb_file tb_data.npz \
  --theta_min -15 --theta_max 15 --ntheta 200 \
  --phi_min -15 --phi_max 15 --nphi 200 \
  --e_min -1.0 --e_max 0.1 --ne 200 \
  --hv 84 --workf 4.5 --v0 12 --temp 10 --polar P \
  --engine grizzly --device cuda --layout full \
  --theta_chunk 20 --e_fermi 0.0
```

---

## Update log

- **2026-08-31:** Paper record (Run A logged 762 s; Run B estimated ~1970 s).
