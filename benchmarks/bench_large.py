#!/usr/bin/env python3
"""Large-grid benchmarks: ME + full pipeline (15³, 20³, 25³)."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chinook.ARPES_lib as arpes_lib  # noqa: E402

from benchmarks._bench_common import (  # noqa: E402
    append_result_row,
    device_label,
    make_graphene_experiment,
    time_chinook_serial_mk,
    time_grizzly_mk,
    valid_peak_indices,
)
from benchmarks.bench_full import (  # noqa: E402
    REPEATS,
    _median,
    _tb_and_dict,
    time_chinook_full,
    time_grizzly_full,
    time_spectral_shared_mk,
)
from grizzly.experiment import GrizzlyExperiment  # noqa: E402

LARGE_GRIDS = {
    "medium": (15, 15, 15),
    "large": (20, 20, 20),
    "xlarge": (25, 25, 25),
}


def append_section(path: Path, title: str, device: str, host: str) -> None:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8") as f:
        f.write(
            f"\n## {title}\n\n"
            f"- when: {when}\n"
            f"- host: {host}\n"
            f"- device: {device} (auto={device_label()})\n"
            f"- grids: 15³, 20³, 25³ graphene spinless\n"
            f"- method: 1 warmup + {REPEATS} timed runs, median\n\n"
            "### Matrix elements\n\n"
            "| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |\n"
            "|-----------|------|--------|-----------|-----------|---------|-------|\n"
        )


def run_me_rows(results: Path, device: str) -> None:
    for label, (nx, ny, ne) in LARGE_GRIDS.items():
        grid = f"{nx}x{ny}x{ne}"
        print(f"  ME {label} ({grid}) ...", flush=True)
        exp = make_graphene_experiment(nx, ny, ne)
        vidx = valid_peak_indices(exp)
        n = len(vidx)
        exp.Mk = exp.Mk * 0
        exp.serial_Mk(vidx)
        from grizzly.engine import compute_all_Mk

        compute_all_Mk(exp, device=device)
        t_ch = time_chinook_serial_mk(exp, vidx)
        t_gr = time_grizzly_mk(exp, device)
        sp = t_ch / t_gr if t_gr > 0 else float("inf")
        row = (
            f"| matrix_elements | {grid} | {n} | "
            f"{t_ch:.4f} | {t_gr:.4f} | {sp:.2f}x | {label} |"
        )
        append_result_row(results, row)
        print(f"    {row}")


def run_full_rows(results: Path, device: str, grids: dict) -> None:
    with results.open("a", encoding="utf-8") as f:
        f.write(
            "\n### Full pipeline\n\n"
            "- chinook: NumPy diag + serial_Mk + spectral\n"
            "- grizzly: Grizzly solve_H + skip chinook ME + Grizzly spectral\n\n"
            "| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |\n"
            "|-----------|------|--------|-----------|-----------|---------|-------|\n"
        )
    for label, (nx, ny, ne) in grids.items():
        grid = f"{nx}x{ny}x{ne}"
        print(f"  full {label} ({grid}) ...", flush=True)
        t_ch = time_chinook_full(nx, ny, ne)
        t_gr = time_grizzly_full(nx, ny, ne, device)
        sp = t_ch / t_gr if t_gr > 0 else float("inf")
        tb, d = _tb_and_dict(nx, ny, ne)
        exp = arpes_lib.experiment(tb, d)
        exp.datacube(diagonalize=False)
        n = len(valid_peak_indices(exp))
        row = (
            f"| full_pipeline | {grid} | {n} | "
            f"{t_ch:.4f} | {t_gr:.4f} | {sp:.2f}x | {label} |"
        )
        append_result_row(results, row)
        print(f"    {row}")

        t_cs, t_gs, n2 = time_spectral_shared_mk(nx, ny, ne, device)
        sp2 = t_cs / t_gs if t_gs > 0 else float("inf")
        row2 = (
            f"| spectral_shared_Mk | {grid} | {n2} | "
            f"{t_cs:.4f} | {t_gs:.4f} | {sp2:.2f}x | spectral only |"
        )
        append_result_row(results, row2)
        print(f"    {row2}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Large grid benchmarks")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    p.add_argument("--host", default="local")
    p.add_argument("--append", default="Large grids")
    p.add_argument(
        "--skip-xlarge-full",
        action="store_true",
        help="Skip 25³ full pipeline (slow on cluster)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = ROOT / "benchmarks" / "RESULTS.md"
    append_section(results, args.append, device, args.host)

    print(f"bench_large — device={device} host={args.host}")
    run_me_rows(results, device)

    full_grids = dict(LARGE_GRIDS)
    if args.skip_xlarge_full:
        full_grids.pop("xlarge", None)
    run_full_rows(results, device, full_grids)

    print(f"\nAppended {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
