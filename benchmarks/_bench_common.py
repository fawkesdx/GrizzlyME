"""Shared helpers for GrizzlyME benchmarks."""

from __future__ import annotations

import collections
import collections.abc
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import chinook.build_lib as build_lib
import chinook.ARPES_lib as arpes_lib

GRIZZLY_ROOT = Path(__file__).resolve().parents[1]
if str(GRIZZLY_ROOT) not in sys.path:
    sys.path.insert(0, str(GRIZZLY_ROOT))

from grizzly.engine import compute_all_Mk
from grizzly.utils import get_device


def make_graphene_experiment(n_kx: int, n_ky: int, n_e: int):
    """Graphene TB + ARPES experiment after datacube() (Mk already from chinook serial)."""
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
    tb_model = build_lib.gen_TB(
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
    exp = arpes_lib.experiment(tb_model, arpes_dict)
    exp.datacube(diagonalize=False)
    return exp


def valid_peak_indices(exp) -> np.ndarray:
    return np.array([i for i in range(len(exp.pks)) if exp.th[i] >= 0])


def time_chinook_serial_mk(exp, valid_indices: np.ndarray, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        exp.Mk = np.zeros_like(exp.Mk)
        t0 = time.perf_counter()
        exp.serial_Mk(valid_indices)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_grizzly_mk(exp, device_str: str, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        compute_all_Mk(exp, device=device_str)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def device_label() -> str:
    dev = get_device("auto")
    if dev.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return str(dev)


def write_results_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# GrizzlyME benchmarks\n\n"
        f"- when: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"- device: cpu (bench forced; auto={device_label()})\n"
        f"- python: {sys.version.split()[0]}\n\n"
        "## Matrix elements (Task 6a)\n\n"
        "Method: 1 warmup + 3 timed runs, report median. "
        "Chinook = `serial_Mk` rerun; Grizzly = `compute_all_Mk`. "
        "Setup `datacube()` not timed.\n\n"
        "| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |\n"
        "|-----------|------|--------|-----------|-----------|---------|-------|\n",
        encoding="utf-8",
    )


def append_result_row(path: Path, row: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(row + "\n")
