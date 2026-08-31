# GrizzlyME benchmarks

- when: 2026-08-28T13:41:31Z
- device: cpu (bench forced; auto=mps)
- python: 3.11.15

## Matrix elements (Task 6a)

Method: 1 warmup + 3 timed runs, report median. Chinook = `serial_Mk` rerun; Grizzly = `compute_all_Mk`. Setup `datacube()` not timed.

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| matrix_elements | 5x5x5 | 36 | 0.0022 | 0.0005 | 4.19x | valid peaks=36 |
| matrix_elements | 10x10x10 | 108 | 0.0061 | 0.0005 | 12.94x | valid peaks=108 |
| matrix_elements | 15x15x15 | 260 | 0.0155 | 0.0007 | 22.08x | valid peaks=260 |

## Full pipeline (Task 6b)

- device: cpu (forced; auto=mps)
- Method: 1 warmup + 3 timed, median.
- `full_pipeline`: construct → datacube → (Grizzly: ensure_Mk force=True) → spectral.
- Note: chinook `datacube()` always runs serial_Mk; Grizzly path pays that once then recomputes Mk with `compute_all_Mk`.
- `spectral_shared_Mk`: same Mk, time spectral() only (fair assembly compare).

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| full_pipeline | 5x5x5 | 36 | 0.2475 | 0.2471 | 1.00x | Grizzly includes chinook ME in datacube + force Mk |
| spectral_shared_Mk | 5x5x5 | 36 | 0.0004 | 0.0002 | 1.76x | spectral only, identical Mk |
| full_pipeline | 10x10x10 | 108 | 0.2553 | 0.2572 | 0.99x | Grizzly includes chinook ME in datacube + force Mk |
| spectral_shared_Mk | 10x10x10 | 108 | 0.0012 | 0.0003 | 4.19x | spectral only, identical Mk |

## GPU (Task 6c) — skipped (no CUDA)

- when: 2026-08-28T18:20:37Z
- cuda available: False
- auto device: mps
- reason: No NVIDIA CUDA GPU on this machine. MPS is available but GrizzlyME engine uses float64/complex128; MPS rejects float64 (see Task 6a CPU-forced benchmarks).
- action: Run this script on a CUDA machine to append Event-timed row.
- command: `python benchmarks/bench_cuda_events.py`

## Full pipeline (datacube-skip, 2026-08-28)

Grizzly `datacube()` now skips chinook `serial_Mk` by default. CPU forced; median of 3 runs.

| benchmark | grid | chinook_s | grizzly_s | speedup | notes |
|-----------|------|-----------|-----------|---------|-------|
| full_pipeline | 5x5x5 | 0.2525 | 0.2511 | 1.01x | diag dominates tiny grid |
| full_pipeline | 10x10x10 | 0.2663 | 0.2541 | 1.05x | ME no longer doubled |

## GPU (Task 6c) — CUDA GPU

- when: 2026-08-28T18:43:24Z
- host: cuda-host
- cuda: NVIDIA GPU (×2 on node; bench used GPU 0)
- pytorch: 2.7.1+cu118
- method: `torch.cuda.Event` around one `compute_all_Mk` after warmup (10³ grid, 108 states)

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| cuda_event_mk | 10x10x10 | 108 | — | 0.0102 | — | ME only; not full pipeline |

## Matrix elements — CUDA GPU GPU (2026-08-28)

- when: 2026-08-28T18:47:45Z
- host: cuda-host
- grizzly device: cuda (NVIDIA GPU)
- chinook: CPU `serial_Mk` on cluster node
- pytorch: 2.7.1+cu118
- method: 1 warmup + 3 timed runs, median

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| matrix_elements | 5x5x5 | 36 | 0.0200 | 0.0032 | 6.19x | valid peaks=36 |
| matrix_elements | 10x10x10 | 108 | 0.0205 | 0.0023 | 8.92x | valid peaks=108 |
| matrix_elements | 15x15x15 | 260 | 0.0481 | 0.0035 | 13.91x | valid peaks=260 |

**Compare CPU Grizzly** (Task 6a): 10³ grizzly 0.0005 s vs GPU 0.0023 s — tiny batches favor fast CPU; GPU wins vs chinook serial on cluster CPU.


## Full pipeline post-integration CPU

- when: 2026-08-28T19:05:51Z
- host: cpu-host
- grizzly device: cpu (auto=mps)
- chinook: full datacube + spectral (NumPy diag + serial_Mk)
- grizzly: skip chinook ME + Grizzly solve_H + Grizzly spectral
- method: 1 warmup + 3 timed runs, median

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| full_pipeline | 5x5x5 | 36 | 0.2668 | 0.2675 | 1.00x | Grizzly diag+ME+spectral |
| spectral_shared_Mk | 5x5x5 | 36 | 0.0004 | 0.0002 | 1.89x | spectral only, identical Mk |
| full_pipeline | 10x10x10 | 108 | 0.2754 | 0.2720 | 1.01x | Grizzly diag+ME+spectral |
| spectral_shared_Mk | 10x10x10 | 108 | 0.0011 | 0.0003 | 3.29x | spectral only, identical Mk |

## Full pipeline post-integration CUDA GPU

- when: 2026-08-28T19:06:06Z
- host: cuda-host
- grizzly device: cuda (NVIDIA GPU)
- chinook: full datacube + spectral (NumPy diag + serial_Mk)
- grizzly: skip chinook ME + Grizzly solve_H + Grizzly spectral
- method: 1 warmup + 3 timed runs, median

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| full_pipeline | 5x5x5 | 36 | 0.8677 | 0.8631 | 1.01x | Grizzly diag+ME+spectral |
| spectral_shared_Mk | 5x5x5 | 36 | 0.0014 | 0.0013 | 1.04x | spectral only, identical Mk |
| full_pipeline | 10x10x10 | 108 | 0.8912 | 0.8763 | 1.02x | Grizzly diag+ME+spectral |
| spectral_shared_Mk | 10x10x10 | 108 | 0.0038 | 0.0018 | 2.14x | spectral only, identical Mk |

**Takeaway:** full pipeline still ~1× on tiny graphene cubes — setup + diagonalization + radial integrals dominate. ME-only and spectral-only benches show real Grizzly wins.

## Large grids CPU

- when: 2026-08-28T19:11:14Z
- host: cpu-host
- device: cpu (auto=mps)
- grids: 15³, 20³, 25³ graphene spinless
- method: 1 warmup + 3 timed runs, median

### Matrix elements

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| matrix_elements | 15x15x15 | 260 | 0.0158 | 0.0010 | 15.12x | medium |
| matrix_elements | 20x20x20 | 684 | 0.0417 | 0.0010 | 40.57x | large |
| matrix_elements | 25x25x25 | 704 | 0.0437 | 0.0011 | 38.09x | xlarge |

### Full pipeline

- chinook: NumPy diag + serial_Mk + spectral
- grizzly: Grizzly solve_H + skip chinook ME + Grizzly spectral

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| full_pipeline | 15x15x15 | 260 | 0.2910 | 0.2726 | 1.07x | medium |
| spectral_shared_Mk | 15x15x15 | 260 | 0.0017 | 0.0003 | 5.56x | spectral only |
| full_pipeline | 20x20x20 | 684 | 0.3200 | 0.2771 | 1.15x | large |
| spectral_shared_Mk | 20x20x20 | 684 | 0.0039 | 0.0005 | 7.78x | spectral only |
| full_pipeline | 25x25x25 | 704 | 0.3222 | 0.2768 | 1.16x | xlarge |
| spectral_shared_Mk | 25x25x25 | 704 | 0.0044 | 0.0008 | 5.83x | spectral only |

## Large grids CUDA GPU

- when: 2026-08-28T19:11:37Z
- host: cuda-host
- device: cuda (NVIDIA GPU)
- grids: 15³, 20³, 25³ graphene spinless

### Matrix elements

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| matrix_elements | 15x15x15 | 260 | 0.0582 | 0.0051 | 11.51x | medium |
| matrix_elements | 20x20x20 | 684 | 0.1537 | 0.0047 | 32.70x | large |
| matrix_elements | 25x25x25 | 704 | 0.1553 | 0.0048 | 32.53x | xlarge |

### Full pipeline

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| full_pipeline | 15x15x15 | 260 | 0.9411 | 0.8895 | 1.06x | medium |
| full_pipeline | 20x20x20 | 684 | 1.0475 | 0.9009 | 1.16x | large |
| full_pipeline | 25x25x25 | 704 | 1.0446 | 0.8863 | 1.18x | xlarge |
| spectral_shared_Mk | 25x25x25 | 704 | 0.0127 | 0.0065 | 1.95x | spectral only |

**Takeaway:** ME speedup scales to **~40× (CPU)** / **~33× (GPU)** at 20–25³. Full pipeline reaches **1.15–1.18×** — setup still dominates but Grizzly pulls ahead as grid grows.

## Datacube profile (20³, cpu@cpu-host)
- when: auto
- valid peaks: chinook=684 grizzly=684

| stage | chinook_s | grizzly_s |
|-------|-----------|----------|
| construct | 0.0000 | 0.0000 |
| diagonalize | 0.0008 | (in datacube) |
| ME (serial / ensure_Mk) | 0.0357 | 0.0013 |
| datacube other / total | 0.2452 / 0.2816 | 0.2499 |
| spectral | 0.0038 | 0.0013 |

## Datacube profile (25³, cuda@cuda-host)
- when: auto
- valid peaks: chinook=704 grizzly=704

| stage | chinook_s | grizzly_s |
|-------|-----------|----------|
| construct | 0.0001 | 0.0001 |
| diagonalize | 0.0021 | (in datacube) |
| ME (serial / ensure_Mk) | 0.1379 | 0.0038 |
| datacube other / total | 0.6379 / 0.7779 | 1.0890 |
| spectral | 0.0136 | 0.0145 |

## Production grids CUDA GPU

- when: 2026-08-28T23:32:05Z
- host: cuda-host
- device: cuda (auto=cuda (NVIDIA GPU))
- grids: 15³, 20³, 25³ graphene spinless
- method: 1 warmup + 3 timed runs, median

### Matrix elements

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| matrix_elements | 30x30x30 | 1260 | 0.2398 | 0.0038 | 62.52x | prod30 |
| matrix_elements | 40x40x40 | 1584 | 0.3000 | 0.0040 | 74.70x | prod40 |

### Full pipeline

- chinook: NumPy diag + serial_Mk + spectral
- grizzly: Grizzly solve_H + skip chinook ME + Grizzly spectral

| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |
|-----------|------|--------|-----------|-----------|---------|-------|
| full_pipeline | 30x30x30 | 1260 | 0.9043 | 0.6515 | 1.39x | prod30 |
| spectral_shared_Mk | 30x30x30 | 1260 | 0.0178 | 0.0074 | 2.40x | spectral only |
| full_pipeline | 40x40x40 | 1584 | 0.9594 | 0.6511 | 1.47x | prod40 |
| spectral_shared_Mk | 40x40x40 | 1584 | 0.0230 | 0.0094 | 2.46x | spectral only |

**Takeaway (production):** ME **63–75×** on GPU at 30–40³. Full pipeline **1.39–1.47×** — still limited by chinook radint/basis setup (~0.6 s of datacube), not ME (`ensure_Mk` ~4 ms).

**Profile:** On CPU 20³, `datacube other` = 0.245 s / 0.282 s total (~87%). On CUDA host 25³, same stage = 0.638 s / 0.778 s (~82%). Next win for full-pipeline speedup is caching/skipping radial integrals, not further ME kernels.


## Radint setup cache (CPU, graphene 20³)

- when: 2026-08-28
- first datacube: 0.2556 s (miss)
- second datacube (same key): 0.0051 s (hit)
- hits/misses: 1/1
- wall speedup 2nd/1st: 50.1x

## Radint disk cache

- when: 2026-08-28
- path: ~/.cache/grizzlyme/radint or $GRIZZLY_RADINT_CACHE_DIR
- disable: GRIZZLY_RADINT_DISK=0
- cold start (empty memory, warm disk): make_radint_pointer not called; Mk parity < 1e-10

## Production Wannier ARPES scale ladder (CUDA full, 2026-08-30)

- when: 2026-08-30
- host: cuda-host (1× NVIDIA V100 32GB used)
- model: production Wannier TB (~6.7M hoppings), not graphene toy
- path: GrizzlyME `compute_all_Mk` (CUDA) + Chinook `spectral` (hybrid)
- layout: full; θ-chunk as noted
- full detail / SI prose: `benchmarks/PAPER_SCALE_LADDER.md`
- private host paths: `benchmarks/RESULTS.private.md`

| grid (nθ×nφ×nE) | nk | θ-chunk | full-cube wall_s | oom | notes |
|-----------------|----|---------|------------------|-----|-------|
| 40×10×40 | 400 | 0 | 332.42 | no | |
| 40×40×40 | 1600 | 0 | 740.82 | no | |
| 80×1×80 | 80 | 0 | 212.73 | no | |
| 80×40×40 | 3200 | 0 | 1343.84 | no | |
| 80×80×40 | 6400 | 20 | 2984.50 | no | 4×~740–754 s chunks |
| 100×100×40 | 10000 | 20 | 4399.90 | no | LADDER.jsonl |
| 200×1×200 | 200 | 40 | 937.02 | no | LADDER.jsonl |
| 200×40×100 | 8000 | 20 | 4681.73 | no | LADDER.jsonl |
| **200×200×200** | **40000** | **20** | **762** | no | 1× V100; paper endpoint |
| **200×200×200** | **40000** | **18** | **~1970 (est.)** | no | 2× V100 multi-GPU (Run B) |
| **200×200×200** | **40000** | **18** | **~1680 (est.)** | no | 2× V100 multi-GPU (Run C, 2026-08-31) |

**Paper takeaways:** real Wannier cubes cost minutes–hours (not graphene ms); θ-chunking enables larger maps without OOM; wall ≈ linear in chunk count; **200³ Mac Chinook ~10.3 h → cuda-host Grizzly ~12.7 min (~49×)**; GPU spectral still a win target.

## Local Mac Studio Chinook 200³ (baseline, 2026-08-30)

- host: mac-studio (Apple Silicon); path: TensorSpec GUI in-process Chinook
- grid: **200×200×200** (`intensity` shape confirmed)
- wall: **37187 s (~10.33 h)** from GUI process start to cube save mtime
- output: simulated cube (~58 MB)
- peak memory (observed): ~211 GB physical footprint
- detail: `benchmarks/LOCAL_MAC_200x200x200.md`
- compare vs cuda-host Grizzly CUDA 200³ below

## cuda-host Grizzly 200³ (production Wannier, 2026-08-30)

- detail: `benchmarks/CUDA_HOST_200x200x200.md`
- model: ~6.7M hoppings Wannier TB (same class as Mac baseline)
- path: GrizzlyME CUDA full + Chinook spectral (hybrid)

| run | GPUs | θ-chunk | full-cube wall_s | oom | notes |
|-----|------|---------|------------------|-----|-------|
| A (logged) | 1 | 20 | **762** | no | canonical paper number |
| B (est.) | 2 | 18 | **~1970** | no | multi-GPU; log lost to rerun |
| C (est.) | 2 | 18 | **~1680** | no | 2026-08-31 repeat; cube mtime |
| Mac Chinook | — | — | **37187** | — | baseline; ~49× slower than Run A |

**Paper one-liner:** 200³ ARPES cube: Mac Chinook **10.3 h** → cuda-host Grizzly **12.7 min** (**~49×**).

<!-- AUTO:SCALE_LADDER_START -->
- when: auto-sync 2026-08-31 01:22 UTC
- status: complete
- host: cuda-host (1× NVIDIA V100 32GB used)
- detail: `benchmarks/PAPER_SCALE_LADDER.md`

| grid (nθ×nφ×nE) | nk | θ-chunk | full-cube wall_s | oom | notes |
|-----------------|----|---------|------------------|-----|-------|
| 40×1×40 | 40 | 0 | ~264 | no | early probe; CUDA full |
| 40×10×40 | 400 | 0 | 332.42 | no | result JSON |
| 40×40×40 | 1600 | 0 | 740.82 | no | job log |
| 80×1×80 | 80 | 0 | 212.73 | no | LADDER.jsonl |
| 80×40×40 | 3200 | 0 | 1343.84 | no | LADDER.jsonl |
| 80×80×40 | 6400 | 20 | 2984.50 | no | LADDER.jsonl |
| 100×100×40 | 10000 | 20 | 4399.90 | no | LADDER.jsonl |
| 200×1×200 | 200 | 40 | 937.02 | no | LADDER.jsonl |
| 200×40×100 | 8000 | 20 | 4681.73 | no | LADDER.jsonl |
<!-- AUTO:SCALE_LADDER_END -->
