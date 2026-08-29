# GrizzlyME

**GPU-accelerated ARPES matrix elements** for [chinook](https://github.com/rpday/chinook), implemented in PyTorch.

Drop-in companion: keep your chinook TB + `ARPES_dict` workflow; GrizzlyME speeds up matrix elements, k-mesh diagonalization, and spectral assembly on **CPU** and **CUDA**.

[Documentation](https://fawkesdx.github.io/GrizzlyME/) · [Source](https://github.com/fawkesdx/GrizzlyME) · [PyPI](https://pypi.org/project/grizzlyme/)

> **v0.1.x alpha — spinless models only.** Spin / SARPES raise a clear error and are planned for a later release.

## Install

```bash
pip install grizzlyme
```

Requires **Python ≥ 3.10**, **PyTorch ≥ 2.0**, and **chinook ≥ 1.1.3** (installed automatically).

```bash
# development clone
pip install -e ".[dev]"
```

**Devices:** `device="auto"` prefers CUDA when available, else CPU. Apple **MPS** is not supported for the float64 engine — use `device="cpu"`.

## Quick start

```python
import chinook.build_lib as build_lib
from grizzly import GrizzlyExperiment

TB = build_lib.gen_TB(basis_dict, hamiltonian_dict)
exp = GrizzlyExperiment(TB, ARPES_dict, device="auto")  # or "cpu" / "cuda"
exp.datacube()
I, Ig = exp.spectral()
```

`I` is the raw intensity cube; `Ig` is resolution-broadened.

## What is accelerated

| Chinook path | GrizzlyME |
|--------------|-----------|
| `serial_Mk` / matrix elements | Batched `compute_all_Mk` |
| `solve_H` over the k-mesh | Batched `solve_H` |
| Spectral assembly + Gaussian broaden | Torch spectral + FFT broaden |

TB construction and (first-call) Slater radial integrals still use chinook. Repeat cubes reuse radials via an in-process / disk cache.

## Performance (indicative)

Vs chinook `serial_Mk` on graphene spinless cubes (see docs for protocol):

| Benchmark | Grid | Speedup |
|-----------|------|---------|
| Matrix elements | ~10³ | ~9–13× |
| Matrix elements | ~40³ | ~75× (CUDA) |
| Full pipeline (first cube) | ~40³ | ~1.5× (setup still chinook-dominated) |

## Scope & limits

**Supported:** spinless TB; momentum (`X,Y,E`) and angle (`Tx,Ty,E`) cubes; CPU and CUDA.

**Not in v0.1:** spin / SARPES; MPS float64; chinook’s interactive Tk map GUI (not required for `datacube` / `spectral`).

Details: [Scope & limits](https://fawkesdx.github.io/GrizzlyME/limits/).

## Name

- **Grizzly** — Grizzly Peak near Berkeley Lab / MAESTRO; institutionally independent; heavy-duty compute
- **ME** — matrix element
- **G.R.I.Z.Z.L.Y.** — GPU Rendered Inner-products for Zone-by-Zone Lattice Yields

[Name & branding](https://fawkesdx.github.io/GrizzlyME/naming/).

## Links

| | |
|--|--|
| Docs | https://fawkesdx.github.io/GrizzlyME/ |
| Issues | https://github.com/fawkesdx/GrizzlyME/issues |
| chinook | https://github.com/rpday/chinook |

## License & citation

MIT — see `LICENSE`. Respect [chinook](https://github.com/rpday/chinook)’s license for that dependency.

If you use GrizzlyME, cite this project and chinook (Day et al., *npj Quantum Materials* 4, 54 (2019)).
