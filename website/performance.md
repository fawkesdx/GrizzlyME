# Performance

Indicative timings vs chinook on graphene spinless cubes. Protocol: warmup + median of timed runs; matrix-element benches exclude one-time setup where noted. Full numbers: `benchmarks/RESULTS.md`.

## Matrix elements

| Grid (approx.) | Speedup vs chinook `serial_Mk` |
|----------------|--------------------------------|
| 5³ | ~4× |
| 10³ | ~9–13× |
| 15³–25³ | ~12–40× |
| 30³–40³ (CUDA) | ~60–75× |

## Full pipeline

End-to-end `datacube` + spectral is often **~1.2–1.5×** because chinook still builds radial integrals and mesh setup on first call. After radint cache hits, repeat cubes are much cheaper.

## Spectral only

With shared `Mk`, Grizzly spectral assembly is typically a few× faster than chinook on mid-size grids.

## How to reproduce

```bash
python benchmarks/bench_me.py
python benchmarks/bench_full.py --append "Full pipeline"
python benchmarks/bench_large.py --append "Large grids"
```

## Production Wannier scale ladder

Indicative **full-cube** walls on a real Wannier TB (~10⁶ hoppings), GrizzlyME CUDA `layout=full` (hybrid Chinook spectral), one V100:

| Grid | Wall | Notes |
|------|------|-------|
| 40×40×40 | ~12 min | no θ-chunk |
| 80×40×40 | ~22 min | no θ-chunk |
| 80×80×40 | ~50 min | θ-chunk=20 |

Larger grids use θ-chunking to avoid OOM. Full table: `benchmarks/RESULTS.md` and `benchmarks/PAPER_SCALE_LADDER.md`.

## Production 200³ endpoint

| Platform | Wall | Notes |
|----------|------|-------|
| Mac Studio Chinook (local) | ~10.3 h | baseline; ~200 GB peak RAM |
| cuda-host Grizzly 1× V100 | **12.7 min (762 s)** | θ-chunk=20; ~49× vs Mac |
| cuda-host Grizzly 2× V100 | ~33 min (est.) | multi-GPU θ-chunk=18 |

Detail: `benchmarks/CUDA_HOST_200x200x200.md`, `benchmarks/LOCAL_MAC_200x200x200.md`.
