# Paper draft notes — production Wannier ARPES scale ladder (GrizzlyME CUDA)

**Status:** living notes for GrizzlyME paper / SI. Do not delete.
**Captured:** 2026-08-30 (ladder still running past 100×100×40 at capture time)
**Purpose:** mid-grid → near-200³ stress ladder on a real Wannier tight-binding model (not graphene toy), measuring wall time, OOM, and θ-chunking.

Public / anonymized twin lives in `RESULTS.md` and `website/performance.md`.
Private hostnames and absolute paths: `benchmarks/RESULTS.private.md` (gitignored).

---

## Hardware / software

- Host: **cuda-host** (cluster GPU node)
- GPU: Tesla V100-SXM2-32GB ×2; ladder jobs used **GPU 0 only** unless noted
- Path: hybrid — GrizzlyME `compute_all_Mk` + Chinook `spectral()`
- Runner: remote ARPES runner (`--engine grizzly --device cuda --layout full`)
- Ladder driver: `bench_grizzly_ladder.py` / `bench_grizzly_scale.py`
- Artifacts: `LADDER.jsonl`, `result_*.json`, `log_*.txt`, `cube_*.npz`

## Physics / model (paper-relevant)

- Production Wannier TB: **~6.7M hoppings** reconstructed on cluster
- Typical ARPES window: θ,φ ∈ [−15°, 15°] (narrower on last planned rung), E ∈ [−1.0, 0.1] eV
- hv=84 eV, workf=4.5, V0=12.0, T=10 K, p-pol
- Correctness prerequisites (must mention if claiming dispersive maps):
  1. Restore emission angles `th`/`ph` after Chinook `datacube()` when using custom `K_BULK` (else all θ→0 → flat Mk)
  2. Exclude exact-zero eigenvalues (`|E| < 1e-10`) after EF shift (uncoupled Wannier orbitals flood E_F)

## Protocol

- Layout: **full** GrizzlyME+kmesh on CUDA (not slices)
- θ-chunk: 0 = single pass; >0 = sequential θ blocks (RAM/VRAM relief)
- Wall clock = log line `full-cube wall: Xs` (compute); `wall_s` in JSON includes process overhead
- OOM = CUDA / host OOM or forced fallback to slices
- Cubes written under `scale_bench/` (does **not** clobber GUI `chinook_arpes_cube.npz`)

## Results table (auto)

<!-- AUTO:RESULTS_START -->
**Auto-updated:** 2026-08-31 01:22 UTC — ladder status: **complete**

| Grid (nθ × nφ × nE) | nk (=nθ×nφ) | θ-chunk | full-cube wall (s) | wall_s (JSON) | OOM | notes |
|---------------------|-------------|---------|--------------------|---------------|-----|-------|
| 40 × 1 × 40 | 40 | 0 | ~264 | — | no | early probe; CUDA full |
| 40 × 10 × 40 | 400 | 0 | 332.42 | — | no | result JSON |
| 40 × 40 × 40 | 1600 | 0 | 740.82 | — | no | job log |
| 80 × 1 × 80 | 80 | 0 | 212.73 | — | no | LADDER.jsonl |
| 80 × 40 × 40 | 3200 | 0 | 1343.84 | 1401.4 | no | LADDER.jsonl |
| 80 × 80 × 40 | 6400 | 20 | 2984.50 | 3043.9 | no | LADDER.jsonl |
| 100 × 100 × 40 | 10000 | 20 | 4399.90 | 4458.6 | no | LADDER.jsonl |
| 200 × 1 × 200 | 200 | 40 | 937.02 | 995.3 | no | LADDER.jsonl |
| 200 × 40 × 100 | 8000 | 20 | 4681.73 | 4740.8 | no | LADDER.jsonl |
| **200 × 200 × 200** | **40000** | **20** | **762** | — | no | Bare ME; 1× V100 (Run A) |
| **200 × 200 × 200** | **40000** | **20** | **~2715** | — | no | **Full ME**; 1× V100 (Run E) |
| **200 × 200 × 200** | **40000** | **18** | **~1970 (est.)** | — | no | GUI run; 2× V100 multi-chunk; log overwritten |

Full 200³ detail: `CUDA_HOST_200x200x200.md`. Mac baseline: `LOCAL_MAC_200x200x200.md` (**37187 s** Chinook).

### 80 × 80 × 40 chunk breakdown

| chunk | θ range (deg) | wall (s) |
|-------|---------------|----------|
| 1/4 | -15.000 … -7.785 | 737.40 |
| 2/4 | -7.405 … -0.190 | 746.40 |
| 3/4 | 0.190 … 7.405 | 746.32 |
| 4/4 | 7.785 … 15.000 | 754.38 |
| **full-cube** | | **2984.50** |

### 100 × 100 × 40 chunk breakdown

| chunk | θ range (deg) | wall (s) |
|-------|---------------|----------|
| 1/5 | -15.000 … -9.242 | 869.80 |
| 2/5 | -8.939 … -3.182 | 884.60 |
| 3/5 | -2.879 … 2.879 | 876.12 |
| 4/5 | 3.182 … 8.939 | 886.89 |
| 5/5 | 9.242 … 15.000 | 882.48 |
| **full-cube** | | **4399.90** |

### 200 × 40 × 100 chunk breakdown

| chunk | θ range (deg) | wall (s) |
|-------|---------------|----------|
| 1/10 | -15.000 … -12.136 | 464.46 |
| 2/10 | -11.985 … -9.121 | 465.96 |
| 3/10 | -8.970 … -6.106 | 465.60 |
| 4/10 | -5.955 … -3.090 | 467.02 |
| 5/10 | -2.940 … -0.075 | 468.14 |
| 6/10 | 0.075 … 2.940 | 467.74 |
| 7/10 | 3.090 … 5.955 | 474.68 |
| 8/10 | 6.106 … 8.970 | 470.64 |
| 9/10 | 9.121 … 11.985 | 468.94 |
| 10/10 | 12.136 … 15.000 | 468.54 |
| **full-cube** | | **4681.73** |

### 200 × 1 × 200 chunk breakdown

| chunk | θ range (deg) | wall (s) |
|-------|---------------|----------|
| 1/5 | -15.000 … -9.121 | 187.18 |
| 2/5 | -8.970 … -3.090 | 187.20 |
| 3/5 | -2.940 … 2.940 | 187.77 |
| 4/5 | 3.090 … 8.970 | 187.79 |
| 5/5 | 9.121 … 15.000 | 187.09 |
| **full-cube** | | **937.02** |
<!-- AUTO:RESULTS_END -->


## Scaling observations (draft prose seeds)

1. **Real Wannier models dominate graphene toy benches.** Graphene ME-only 40³ is milliseconds; this TB’s full ARPES cube at 80×40×40 is ~22 min on one V100.
2. **θ-chunking is the practical memory lever** before multi-GPU. 80×80×40 needed chunk=20; no OOM.
3. **Wall ≈ linear in number of equal θ-chunks** at fixed (nφ, nE) — chunk times nearly flat (~12–15 min).
4. **nk alone is not enough:** 80×1×80 (nk=80) finishes in ~3.5 min; 80×40×40 (nk=3200) ~22 min — φ×orbital×ME cost matters.
5. **Hybrid path:** Grizzly ME on CUDA + Chinook spectral still CPU-bound on spectral assembly for large cubes → paper Module-3 motivation (GPU spectral).
6. **Dual V100:** Full ME multi-GPU ~28–33 min (Runs B–C); Bare single-GPU **762 s** still fastest absolute wall.
7. **200³ vs Mac:** Bare ME → **12.7 min (~49×)**; Full ME 1×GPU → **~45 min (~14×)**; see `CUDA_HOST_200x200x200.md`.

## Related early path findings (include if discussing “why CUDA full”)

| Mode | Observation |
|------|-------------|
| Local Chinook | Fastest for *small* maps |
| Grizzly CPU + slices | Throttled workers; often slower than local Chinook |
| Grizzly CUDA + slices | Cap ~2 workers — poor |
| Grizzly CUDA + full | Best remote path for mid/large grids |

## Reproduce

```bash
# On cuda-host, in run directory with TB + physics staged:
python -u bench_grizzly_ladder.py --out_dir scale_bench
# or single rung:
python -u bench_grizzly_scale.py --ntheta 80 --nphi 80 --ne 40 \
  --device cuda --layout full --theta_chunk 20 --out_dir scale_bench
```

## Local Mac 200³ baseline (compare later)

**COMPLETED.** See `LOCAL_MAC_200x200x200.md`.

| | |
|--|--|
| Grid | 200×200×200 Chinook local (Mac Studio GUI) |
| Wall (GUI start → cube mtime) | **37187 s (~10.33 h)** |
| Output | simulated cube (~61 MB; intensity shape confirmed) |
| Peak RAM (observe) | ~211 GB footprint |

**cuda-host 200³ Grizzly recorded** — see `CUDA_HOST_200x200x200.md` (Bare 762 s; Full ME ~2715 s 1×GPU; ~1680–1970 s est. dual-GPU).

## Update log

- 2026-08-31: cuda-host full **200×200×200** Grizzly runs recorded (`CUDA_HOST_200x200x200.md`); Mac compare updated.
- 2026-08-30: initial capture from cuda-host scale ladder + agent session; ladder not finished.
- Auto-sync: private script rewrites `<!-- AUTO:RESULTS -->` from cluster until ladder complete (not in public repo).
- 2026-08-30: started watch on local Mac 200³ → `LOCAL_MAC_200x200x200.md`.

<!-- AUTO:UPDATE_LOG_START -->
- 2026-08-31 01:22 UTC: auto-sync (complete)
<!-- AUTO:UPDATE_LOG_END -->

<!-- AUTO:UPDATE_LOG_START -->
- 2026-08-30 15:05 UTC: auto-sync (running (100x100x40))
<!-- AUTO:UPDATE_LOG_END -->
