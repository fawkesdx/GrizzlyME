# Usage

Build the TB model and ARPES dictionary exactly as you would for chinook, then wrap with `GrizzlyExperiment`:

```python
import chinook.build_lib as build_lib
from grizzly import GrizzlyExperiment

TB = build_lib.gen_TB(basis_dict, hamiltonian_dict)

exp = GrizzlyExperiment(TB, ARPES_dict, device="auto")
exp.datacube()
I, Ig = exp.spectral()
```

- `I` — raw intensity cube  
- `Ig` — resolution-broadened cube  

## Defaults

`datacube()` by default:

- Skips chinook’s serial matrix-element loop and fills `Mk` with Grizzly
- Uses Grizzly `solve_H` during k-mesh diagonalization

```python
exp.datacube(
    skip_chinook_mk=True,
    use_grizzly_diagonalize=True,
)
```

## Radial integrals

Through `prepare_me_shell`, `rad_type="slater"` (the default) and
`rad_type="hydrogenic"` are computed as a batched torch quadrature since v0.2.0;
`grid`, `exec` and `fixed` fall back to chinook. TensorSpec Full-ME uses this shell
path. Stock `GrizzlyExperiment.datacube()` still uses chinook for its cold radial
build until wired to `prepare_me_shell`. Force a shell backend with
`GRIZZLY_RADINT=auto|torch|chinook` or the `radint_mode` argument of
`prepare_me_shell`.

Radial-integral setup is cached after the first successful build:

- **Memory** — same process
- **Disk** — `~/.cache/grizzlyme/radint/` (override with `$GRIZZLY_RADINT_CACHE_DIR`)

Disable disk: `GRIZZLY_RADINT_DISK=0`.

Changing photon energy, work function, basis, `rad_type`, or energy window invalidates the key and rebuilds.

## Lower-level API

```python
from grizzly import compute_all_Mk, solve_H

Mk = compute_all_Mk(exp, device="cuda")
Eband, Evec = solve_H(TB, device="cuda")
```
