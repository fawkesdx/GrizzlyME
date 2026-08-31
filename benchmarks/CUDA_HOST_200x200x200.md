# CUDA cluster — 200×200×200 GrizzlyME CUDA ARPES (paper record)

**Do not delete.** Production-scale endpoint for GrizzlyME vs local Mac Chinook baseline.
**Status:** Run A = Bare ME (fast); **Run E = Full ME 1× GPU** (production physics, logged via process+cube).

Public summary: `RESULTS.md` · Mac baseline: `LOCAL_MAC_200x200x200.md` · Mid-grid ladder: `PAPER_SCALE_LADDER.md`.

---

## Job identity (all runs)

| Field | Value |
|-------|--------|
| Host | cuda-host (1× or 2× NVIDIA V100 32GB) |
| TB | Production Wannier `tb_data.npz` (~107 MB; ~6.7M hoppings, 408 bands) |
| Physics | hv=84 eV, workf=4.5, V0=12, T=10 K, p-pol, hkl=(-2,0,1) |
| Grid | **200 × 200 × 200** (θ × φ × E), θ,φ ∈ [−15°, 15°] |
| Engine | GrizzlyME CUDA, layout **full**, hybrid path (Grizzly ME + Chinook spectral) |
| Runner | remote ARPES runner (TensorSpec GUI / SSH dispatch) |
| Output | `chinook_arpes_cube.npz` |

**ME modes matter:** Bare (spectral only) skips Chinook `datacube`/Mk → ~13 min. Full ME pays ~200 s `datacube` per θ-chunk → ~45 min on 1× GPU with current code.

---

## Run A — Bare ME, single-GPU (fast path; not Full ME)

| Metric | Value |
|--------|-------|
| **full-cube wall** | **762 s (~12.7 min)** — log `full-cube wall: 762.96s` |
| Finished | 2026-08-30 ~20:37 |
| ME | **Bare Spectral Function (ME Off)** |
| GPUs | **1** |
| θ-chunk | **20** (10 blocks) |
| E window | [−1.0, 0.1] eV |
| OOM | no |
| Cube size | ~28 MB |
| Notes | Headline ~49× vs Mac; **physics ≠ Full ME** |

---

## Run B — Full ME, dual-GPU (wall estimated)

| Metric | Value |
|--------|-------|
| **wall (estimated)** | **~1970 s (~32.8 min)** |
| Estimate method | Process elapsed + cube mtime (log overwritten) |
| Finished | 2026-08-30 ~22:44 |
| ME | Full Matrix Elements |
| GPUs | **2** (`Using GPU ids: [0, 1]`) |
| Layout | MULTI-GPU θ-chunk=**18**, **12 blocks** |
| OOM | no |
| Cube size | ~27.8 MB |

---

## Run C — Full ME, dual-GPU repeat (wall estimated)

| Metric | Value |
|--------|-------|
| **wall (estimated)** | **~1680 s (~28.0 min)** |
| Estimate method | Cube mtime (~03:30 → 03:58 PT) |
| Finished | 2026-08-31 03:58 |
| ME | Full Matrix Elements |
| GPUs | **2** |
| θ-chunk | **18**, 12 blocks |
| OOM | no |

---

## Run D — Full ME, dual-GPU (superseded; no logged wall)

| Metric | Value |
|--------|-------|
| Started | 2026-08-31 ~04:04 |
| GPUs | **2**, θ-chunk=**20**, 10 blocks |
| Per-block (early) | ~245–285 s |
| Notes | Log overwritten; cube replaced by Run E |

---

## Run E — Full ME, single-GPU (production physics; 2026-08-31)

| Metric | Value |
|--------|-------|
| **wall (process → cube)** | **~2715 s (~45.2 min)** |
| Estimate method | Dispatch ~04:42 PT → cube mtime **05:27:14**; log overwritten by later submit |
| Per-block (logged early) | **~281 s** (setup~75 + datacube~198 + mk~8) × 10 → ~2810 s cube-only |
| Finished | 2026-08-31 05:27 |
| ME | **Full Matrix Elements** |
| GPUs | **1** (`--ngpus 1`, `Using GPU ids: [0]`) |
| Path | In-process `FULL layout (θ-chunk=20)` — **not** multi-GPU spawn |
| θ-chunk | **20** (10 blocks) |
| E window | **[−1.5, 0.2] eV** |
| OOM | no |
| Cube | shape **(200,200,200)** float32; **29,550,306 B**; `theta_chunk=20`, `engine=GrizzlyME`, `device=cuda` |
| Notes | Bottleneck = Chinook datacube/radint **per θ-chunk** (~70% of block). Tier 0 (`ngpus 1`) correct; Tier 1 (hoist datacube) still needed for ~10–15 min Full ME |

---

## Mac vs cuda-host (paper compare)

| | Mac Studio Chinook | cuda-host Bare (Run A) | cuda-host Full ME 1×GPU (Run E) | cuda-host Full ME 2×GPU (Run C est.) |
|--|-------------------|------------------------|----------------------------------|--------------------------------------|
| wall_s | **37,187 (~10.33 h)** | **762 (~12.7 min)** | **~2,715 (~45.2 min)** | **~1,680 (~28.0 min)** |
| speedup vs Mac | 1× | **~49×** | **~14×** | **~22×** |
| ME | Full (local) | **Bare** | **Full** | **Full** |
| device | Apple Silicon CPU | 1× V100 | 1× V100 | 2× V100 |

**Paper one-liners (draft):**
- Bare ME: Mac Chinook **~10.3 h** → cuda-host Grizzly **12.7 min** (**~49×**).
- Full ME (fair vs Mac Full): Mac **~10.3 h** → cuda-host Grizzly 1× V100 **~45 min** (**~14×**); 2× V100 multi-GPU ~28 min (**~22×**) but spawn path still pays datacube twice in parallel.

---

## Reproduce (generic)

```bash
# Full ME production (1 GPU recommended until datacube hoist lands):
python -u chinook_remote_runner.py \
  --tb_file tb_data.npz \
  --theta_min -15 --theta_max 15 --ntheta 200 \
  --phi_min -15 --phi_max 15 --nphi 200 \
  --e_min -1.5 --e_max 0.2 --ne 200 \
  --hv 84 --workf 4.5 --v0 12 --temp 10 --polar P \
  --engine grizzly --device cuda --layout full \
  --theta_chunk 0 --ngpus 1 --e_fermi 0.0
```

---

## Update log

- **2026-08-31:** Clarified Run A = Bare ME. Added Run E Full ME 1×GPU ~2715 s (45.2 min). Runs B–D Full ME dual-GPU context.
