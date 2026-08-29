# Install

```bash
pip install grizzlyme
```

From GitHub:

```bash
pip install git+https://github.com/fawkesdx/GrizzlyME.git
```

From a clone of this repository:

```bash
pip install -e ".[dev]"
```

Maintainers: [Publishing](publishing.md) (TestPyPI → PyPI).

## Requirements

- Python **3.10–3.12** recommended (3.13 works if chinook’s `pkg_resources` dependency is satisfied)
- [PyTorch](https://pytorch.org/) ≥ 2.0
- [chinook](https://github.com/rpday/chinook) ≥ 1.1.3
- NumPy, psutil
- `setuptools>=61,<82` (pulled in automatically; chinook needs `pkg_resources`)

Optional: `pip install -e ".[bench]"` for matplotlib-based figure scripts.

## Devices

| `device=` | Behavior |
|-----------|----------|
| `"auto"` | Prefer CUDA if available, else CPU |
| `"cuda"` | CUDA GPU |
| `"cpu"` | CPU |

**MPS** (Metal) is detected but **not supported** for the float64 / complex128 engine. Use `device="cpu"` on those systems until a float32 path exists.

Tk / interactive chinook map GUI is **not required**. GrizzlyME stubs `chinook.Tk_plot` when `_tkinter` is missing (common with Homebrew Python).
