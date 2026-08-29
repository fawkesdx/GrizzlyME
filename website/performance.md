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
