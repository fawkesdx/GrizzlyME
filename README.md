# GrizzlyME

GPU-accelerated batched matrix elements for [chinook](https://github.com/rpday/chinook) ARPES simulations (PyTorch).

GrizzlyME is a high-performance companion to chinook: same spinless ARPES workflow, with the matrix-element and spectral hot path rewritten as batched PyTorch ops for CPU and CUDA. Aimed at public install for the ARPES community (GitHub now; PyPI/Conda planned).

**Status:** v0.1 alpha — spinless models. Docs: [fawkesdx.github.io/GrizzlyME](https://fawkesdx.github.io/GrizzlyME/).

## Name

- **Grizzly** — homage to Grizzly Peak behind Berkeley Lab, near MAESTRO; institutionally independent; evokes heavy-duty compute
- **ME** — matrix element
- **G.R.I.Z.Z.L.Y.** — GPU Rendered Inner-products for Zone-by-Zone Lattice Yields

Full note: [docs → Name & branding](https://fawkesdx.github.io/GrizzlyME/naming/).

## Install

```bash
pip install grizzlyme
# or from source:
pip install -e ".[dev]"
```

Requires Python ≥3.10, PyTorch, and chinook (≥1.1.3).

**Devices:** `device='auto'` prefers CUDA when present. Apple **MPS** is not supported for the float64/complex128 engine — use `device='cpu'` on those platforms.

## Quick start

```python
import chinook.build_lib as build_lib
from grizzly import GrizzlyExperiment

TB = build_lib.gen_TB(basis_dict, hamiltonian_dict)
exp = GrizzlyExperiment(TB, ARPES_dict, device="auto")  # or "cpu" / "cuda"
exp.datacube()
I, Ig = exp.spectral()
```

## Performance (indicative)

Graphene spinless cubes vs chinook `serial_Mk` (median of timed runs; see docs for protocol):

| Benchmark | Grid | Speedup |
|-----------|------|---------|
| Matrix elements | 10³ | ~9–13× |
| Matrix elements | 40³ | ~75× (CUDA) |
| Full pipeline | 40³ | ~1.5× (setup still chinook-dominated) |

Repeat `datacube()` on the same model reuses Slater radial integrals (memory + disk cache).

## Tests

```bash
pytest tests/ -q
```

## Documentation

- Site: https://fawkesdx.github.io/GrizzlyME/
- Design notes in-repo under `docs/`

## License

MIT — see `LICENSE`. Depends on chinook; follow chinook’s license for that dependency.

## Citation

If you use GrizzlyME, cite this repository and [chinook](https://github.com/rpday/chinook).
