# Publishing (maintainers)

Package name on PyPI: **`grizzlyme`** (current: **0.1.4** alpha).

## One-time setup

1. Create accounts: [pypi.org](https://pypi.org/account/register/) and [test.pypi.org](https://test.pypi.org/account/register/).
2. Create an API token on each (Account → API tokens).
3. Export credentials (never commit tokens):

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD='pypi-...'   # use TestPyPI token first, then PyPI token for prod
```

## Build

From repository root:

```bash
pip install build twine
python -m build
```

Artifacts: `dist/grizzlyme-0.1.4.tar.gz` and `dist/grizzlyme-0.1.4-py3-none-any.whl`.

## Step 1 — TestPyPI

```bash
twine upload --repository testpypi dist/grizzlyme-0.1.4*
```

Smoke test in a **new venv**:

```bash
python -m venv /tmp/grizzly-test && source /tmp/grizzly-test/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  grizzlyme==0.1.4
python -c "from grizzly import GrizzlyExperiment; print('ok')"
```

The extra index URL is required so dependencies (`torch`, `chinook`, …) install from main PyPI.

## Step 2 — PyPI

```bash
twine upload dist/grizzlyme-0.1.4*
```

Users:

```bash
pip install grizzlyme
```

## Version policy

Bump `version` in `pyproject.toml` for each release. PyPI does not allow replacing an uploaded version.

## 0.1.4 expectations

GPU spectral path: `build_raw_I_from_experiment(..., device=)` and `spectral_maps_from_experiment()` for CUDA/MPS spectral cubes. Document on the PyPI project page: **alpha**, spinless only, requires chinook + PyTorch.
