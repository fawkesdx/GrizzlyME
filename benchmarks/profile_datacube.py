#!/usr/bin/env python3
"""Profile chinook datacube stages vs GrizzlyME hot path.

Run from GrizzlyME root:
    python benchmarks/profile_datacube.py [--n 20] [--device cpu]
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import sys
import time
from pathlib import Path

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chinook.ARPES_lib as arpes_lib  # noqa: E402

from benchmarks._bench_common import (  # noqa: E402
    make_graphene_experiment,
    valid_peak_indices,
)
from benchmarks.bench_full import _tb_and_dict  # noqa: E402
from grizzly.engine import compute_all_Mk  # noqa: E402
from grizzly.experiment import GrizzlyExperiment  # noqa: E402
from grizzly.hamiltonian import solve_H  # noqa: E402
from grizzly.utils import to_numpy  # noqa: E402


def _timed(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:40s} {dt:8.4f} s")
    return out, dt


def profile_chinook(n: int) -> dict:
    print(f"\n=== Chinook profile ({n}³) ===")
    times = {}
    tb, d = _tb_and_dict(n, n, n)

    def make_exp():
        return arpes_lib.experiment(tb, d)

    exp, times["construct"] = _timed("construct experiment", make_exp)

    # Patch stages inside diagonalize/datacube by wrapping methods
    orig_diag = exp.diagonalize
    orig_serial = exp.serial_Mk

    diag_t = [0.0]
    me_t = [0.0]

    def timed_diag(diagonalize=False):
        t0 = time.perf_counter()
        out = orig_diag(diagonalize)
        diag_t[0] = time.perf_counter() - t0
        return out

    def timed_serial(indices):
        t0 = time.perf_counter()
        out = orig_serial(indices)
        me_t[0] = time.perf_counter() - t0
        return out

    exp.diagonalize = timed_diag
    exp.serial_Mk = timed_serial

    t0 = time.perf_counter()
    exp.datacube(diagonalize=False)
    total = time.perf_counter() - t0
    times["diagonalize"] = diag_t[0]
    times["serial_Mk"] = me_t[0]
    times["datacube_other"] = total - diag_t[0] - me_t[0]
    times["datacube_total"] = total
    print(f"  {'diagonalize (inside datacube)':40s} {diag_t[0]:8.4f} s")
    print(f"  {'serial_Mk (inside datacube)':40s} {me_t[0]:8.4f} s")
    print(f"  {'datacube other (radint/basis/pks)':40s} {times['datacube_other']:8.4f} s")
    print(f"  {'datacube TOTAL':40s} {total:8.4f} s")

    _, times["spectral"] = _timed("spectral()", lambda: exp.spectral())
    times["valid_peaks"] = int(np_sum_valid(exp))
    return times


def np_sum_valid(exp) -> int:
    return int(sum(1 for i in range(len(exp.th)) if exp.th[i] >= 0))


def profile_grizzly(n: int, device: str) -> dict:
    print(f"\n=== Grizzly profile ({n}³, device={device}) ===")
    times = {}
    tb, d = _tb_and_dict(n, n, n)

    g, times["construct"] = _timed(
        "construct GrizzlyExperiment",
        lambda: GrizzlyExperiment(tb, d, device=device),
    )

    # Time solve_H alone after a chinook Kobj exists
    exp_setup = arpes_lib.experiment(tb, d)
    # Minimal: run diagonalize only via chinook to set Kobj, then time Grizzly solve_H
    # Better: time full Grizzly datacube and ME separately

    t0 = time.perf_counter()
    g.datacube(skip_chinook_mk=True, use_grizzly_diagonalize=True)
    times["datacube_grizzly"] = time.perf_counter() - t0
    print(f"  {'datacube (Grizzly diag+ME)':40s} {times['datacube_grizzly']:8.4f} s")

    # Re-time ME alone after setup already done (force recompute)
    _, times["ensure_Mk"] = _timed(
        "ensure_Mk(force=True)",
        lambda: g.ensure_Mk(force=True),
    )
    _, times["spectral"] = _timed("spectral()", lambda: g.spectral())
    times["valid_peaks"] = np_sum_valid(g._exp)
    return times


def write_report(path: Path, n: int, device: str, chinook: dict, grizzly: dict) -> None:
    lines = [
        f"\n## Datacube profile ({n}³, {device})\n",
        f"- when: auto\n",
        f"- valid peaks: chinook={chinook.get('valid_peaks')} grizzly={grizzly.get('valid_peaks')}\n",
        "\n| stage | chinook_s | grizzly_s |\n",
        "|-------|-----------|----------|\n",
        f"| construct | {chinook['construct']:.4f} | {grizzly['construct']:.4f} |\n",
        f"| diagonalize | {chinook['diagonalize']:.4f} | (in datacube) |\n",
        f"| ME (serial / ensure_Mk) | {chinook['serial_Mk']:.4f} | {grizzly['ensure_Mk']:.4f} |\n",
        f"| datacube other / total | {chinook['datacube_other']:.4f} / {chinook['datacube_total']:.4f} | {grizzly['datacube_grizzly']:.4f} |\n",
        f"| spectral | {chinook['spectral']:.4f} | {grizzly['spectral']:.4f} |\n",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nAppended profile to {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="Cube edge length")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    p.add_argument("--host", default="local")
    args = p.parse_args()
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = ROOT / "benchmarks" / "RESULTS.md"
    chinook = profile_chinook(args.n)
    grizzly = profile_grizzly(args.n, device)
    write_report(results, args.n, f"{device}@{args.host}", chinook, grizzly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
