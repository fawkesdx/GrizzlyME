#!/usr/bin/env python3
"""Task 6b: full pipeline wall time — chinook vs GrizzlyExperiment."""

from __future__ import annotations

import argparse
import collections
import collections.abc
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chinook.build_lib as build_lib  # noqa: E402
import chinook.ARPES_lib as arpes_lib  # noqa: E402

from benchmarks._bench_common import append_result_row, device_label  # noqa: E402
from grizzly.experiment import GrizzlyExperiment  # noqa: E402

GRIDS = {
    "tiny": (5, 5, 5),
    "small": (10, 10, 10),
}
REPEATS = 3
DEVICE = "cpu"  # float64 parity; MPS unsupported in engine


def _tb_and_dict(n_kx: int, n_ky: int, n_e: int):
    basis_dict = build_lib.gen_basis(
        {
            "atoms": [0, 1],
            "Z": {0: 6, 1: 6},
            "pos": [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.42, 0.0])],
            "orbs": [["21z"], ["21z"]],
            "spin": {"bool": False},
        }
    )
    a_lattice = np.array(
        [[2.46, 0.0, 0.0], [-1.23, 2.130446, 0.0], [0.0, 0.0, 20.0]]
    )
    hopping_list = [
        [0, 0, 0.0, 0.0, 0.0, 0.0],
        [1, 1, 0.0, 0.0, 0.0, 0.0],
        [0, 1, 0.0, 1.42, 0.0, -2.8],
        [0, 1, 1.23, -0.71, 0.0, -2.8],
        [0, 1, -1.23, -0.71, 0.0, -2.8],
    ]
    tb = build_lib.gen_TB(
        basis_dict,
        {
            "type": "list",
            "list": hopping_list,
            "a": a_lattice.tolist(),
            "cutoff": 5.0,
            "spin": {"bool": False},
        },
    )
    arpes_dict = {
        "cube": {
            "Tx": (-15.0, 15.0, n_kx),
            "Ty": (-15.0, 15.0, n_ky),
            "E": (-3.0, 3.0, n_e),
        },
        "hv": 21.2,
        "W": 4.5,
        "pol": np.array([1.0, 0.0, 0.0]),
        "T": 300.0,
        "resolution": {"E": 0.05, "k": 0.02},
        "SE": ["constant", 0.1],
    }
    return tb, arpes_dict


def _median(fn, repeats: int = REPEATS) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_chinook_full(n_kx: int, n_ky: int, n_e: int) -> float:
    def once():
        tb, d = _tb_and_dict(n_kx, n_ky, n_e)
        exp = arpes_lib.experiment(tb, d)
        exp.datacube(diagonalize=False)
        exp.spectral()

    # warmup
    once()
    return _median(once)


def time_grizzly_full(n_kx: int, n_ky: int, n_e: int, device: str) -> float:
    def once():
        tb, d = _tb_and_dict(n_kx, n_ky, n_e)
        g = GrizzlyExperiment(tb, d, device=device)
        g.datacube(
            diagonalize=False,
            skip_chinook_mk=True,
            use_grizzly_diagonalize=True,
        )
        g.spectral()

    once()
    return _median(once)


def time_spectral_shared_mk(
    n_kx: int, n_ky: int, n_e: int, device: str
) -> tuple[float, float, int]:
    """Fair spectral-only: same Mk on both; time spectral() only."""
    tb, d = _tb_and_dict(n_kx, n_ky, n_e)
    exp = arpes_lib.experiment(tb, d)
    exp.datacube(diagonalize=False)
    mk = exp.Mk.copy()
    n_states = int(np.sum(exp.th >= 0))

    g = GrizzlyExperiment(tb, d, device=device)
    g.datacube(diagonalize=False)
    g._exp.Mk = mk.copy()

    def chinook_spec():
        exp.Mk = mk.copy()
        exp.spectral()

    def grizzly_spec():
        g._exp.Mk = mk.copy()
        g.spectral()

    chinook_spec()
    grizzly_spec()
    return _median(chinook_spec), _median(grizzly_spec), n_states


def append_section(path: Path, device: str, title: str, host: str = "local") -> None:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8") as f:
        f.write(
            f"\n## {title}\n\n"
            f"- when: {when}\n"
            f"- host: {host}\n"
            f"- grizzly device: {device} (auto={device_label()})\n"
            f"- chinook: full datacube + spectral (NumPy diag + serial_Mk)\n"
            f"- grizzly: skip chinook ME + Grizzly solve_H + Grizzly spectral\n"
            f"- method: 1 warmup + {REPEATS} timed runs, median\n\n"
            "| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |\n"
            "|-----------|------|--------|-----------|-----------|---------|-------|\n"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full pipeline benchmark")
    p.add_argument("--device", default=DEVICE, choices=("cpu", "cuda", "auto"))
    p.add_argument("--append", metavar="SECTION", help="Append section to RESULTS.md")
    p.add_argument("--host", default="local", help="Host label for RESULTS.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = ROOT / "benchmarks" / "RESULTS.md"
    if not results.exists():
        results.write_text("# GrizzlyME benchmarks\n\n", encoding="utf-8")

    section = args.append or "Full pipeline (Task 6b)"
    append_section(results, device, section, host=args.host)

    print(f"GrizzlyME bench_full — device={device} host={args.host}")

    for label, (nx, ny, ne) in GRIDS.items():
        grid = f"{nx}x{ny}x{ne}"
        print(f"  full_pipeline {label} ...", flush=True)
        t_ch = time_chinook_full(nx, ny, ne)
        t_gr = time_grizzly_full(nx, ny, ne, device)
        sp = t_ch / t_gr if t_gr > 0 else float("inf")
        # states from a quick build
        tb, d = _tb_and_dict(nx, ny, ne)
        exp = arpes_lib.experiment(tb, d)
        exp.datacube(diagonalize=False)
        n_states = int(np.sum(exp.th >= 0))
        row = (
            f"| full_pipeline | {grid} | {n_states} | "
            f"{t_ch:.4f} | {t_gr:.4f} | {sp:.2f}x | "
            f"Grizzly diag+ME+spectral |"
        )
        append_result_row(results, row)
        print(f"    {row}")

        print(f"  spectral_shared_Mk {label} ...", flush=True)
        t_cs, t_gs, n_states = time_spectral_shared_mk(nx, ny, ne, device)
        sp2 = t_cs / t_gs if t_gs > 0 else float("inf")
        row2 = (
            f"| spectral_shared_Mk | {grid} | {n_states} | "
            f"{t_cs:.4f} | {t_gs:.4f} | {sp2:.2f}x | spectral only, identical Mk |"
        )
        append_result_row(results, row2)
        print(f"    {row2}")

    print(f"\nAppended {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
