# GrizzlyME

**GPU-accelerated ARPES matrix elements** for [chinook](https://github.com/rpday/chinook), implemented in PyTorch.

Drop-in companion: keep your chinook TB + `ARPES_dict` workflow; GrizzlyME speeds up matrix elements, k-mesh diagonalization, and spectral assembly on **CPU** and **CUDA**.

[Documentation](https://fawkesdx.github.io/GrizzlyME/) · [Source](https://github.com/fawkesdx/GrizzlyME) · [PyPI](https://pypi.org/project/grizzlyme/)

> **v0.2.x alpha — spinless models only.** Spin / SARPES raise a clear error and are planned for a later release.

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
| `make_radint_pointer` (`slater`, `hydrogenic`) | Batched torch quadrature |

TB construction still uses chinook. The torch radial-integral path is exposed through
`prepare_me_shell` (including TensorSpec Full-ME callers). Stock
`GrizzlyExperiment.datacube()` still builds a cold radial table with chinook until that
API is wired to `prepare_me_shell`; repeat cubes reuse radials via an in-process / disk
cache.

For multi-GPU jobs, prepare the matrix-element shell once in the parent process before spawning workers. The disk cache uses a process-safe lock, so concurrent workers share one radial-integral build instead of duplicating it.

## Radial integrals

Since **v0.2.0**, `prepare_me_shell` can run analytic radial integrals as a
batched torch quadrature instead of chinook's per-energy adaptive recursion. This
removes the multi-minute cold-shell setup cost for TensorSpec Full-ME and other
`prepare_me_shell` callers. Stock `GrizzlyExperiment.datacube()` still uses chinook
for its cold radial build until it is wired to this path. `rad_type="slater"` (the
default) and `rad_type="hydrogenic"` are covered and match chinook to `rtol=1e-5`;
`grid`, `exec` and `fixed` still use chinook.

Select the backend with `radint_mode` or the `GRIZZLY_RADINT` environment variable:

| Value | Behaviour |
|-------|-----------|
| `auto` (default) | torch, falling back to chinook with a `RuntimeWarning` |
| `torch` | torch only; failures propagate |
| `chinook` | chinook only |

```python
from grizzly import prepare_me_shell

shell = prepare_me_shell(exp, radint_mode="auto")
print(shell.radint_source)  # 'torch', 'chinook' or 'cache'
```

A forced backend never reuses a cache entry built by the other one, so parity checks stay honest.

## Spherical-harmonic / Gaunt setup (`all_Y`)

Since **v0.2.1**, `prepare_me_shell` builds the unique `(l,m)` Gaunt tables with
`ylm_torch.batched_gaunt` (parity vs chinook `all_Y` within `1e-10`). Select with
`ylm_mode` or `$GRIZZLY_YLM` (`auto` / `torch` / `chinook`). Angle Ylm during ME
evaluation was already torch (`engine.compute_ylm_batch`).

```python
shell = prepare_me_shell(exp, ylm_mode="auto")
print(shell.ylm_source)  # 'torch' or 'chinook'
```

## Performance (indicative)

Vs chinook `serial_Mk` on graphene spinless cubes (see docs for protocol):

| Benchmark | Grid | Speedup |
|-----------|------|---------|
| Matrix elements | ~10³ | ~9–13× |
| Matrix elements | ~40³ | ~75× (CUDA) |
| Full pipeline (first cube) | ~40³ | ~1.5× (setup still chinook-dominated) |

## Scope & limits

**Supported:** spinless TB; momentum (`X,Y,E`) and angle (`Tx,Ty,E`) cubes; CPU and CUDA.

**Not in v0.2:** spin / SARPES; MPS float64; torch radial integrals for `grid` / `exec` / `fixed` `rad_type`; chinook’s interactive Tk map GUI (not required for `datacube` / `spectral`).

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
