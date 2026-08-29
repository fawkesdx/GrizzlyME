#!/usr/bin/env python3
"""Task 6a: benchmark chinook serial_Mk vs grizzly compute_all_Mk."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Run from GrizzlyME root or benchmarks/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks._bench_common import (  # noqa: E402
    append_result_row,
    device_label,
    make_graphene_experiment,
    time_chinook_serial_mk,
    time_grizzly_mk,
    valid_peak_indices,
    write_results_header,
)
from grizzly.engine import compute_all_Mk  # noqa: E402

GRIDS = {
    "tiny": (5, 5, 5),
    "small": (10, 10, 10),
    "medium": (15, 15, 15),
}


def run_grid(label: str, n_kx: int, n_ky: int, n_e: int, device_str: str) -> str:
    print(f"  grid {label} ({n_kx}x{n_ky}x{n_e}) ...", flush=True)
    exp = make_graphene_experiment(n_kx, n_ky, n_e)
    vidx = valid_peak_indices(exp)
    n_states = len(vidx)
    grid_str = f"{n_kx}x{n_ky}x{n_e}"

    # warmup (untimed)
    exp.Mk = exp.Mk * 0
    exp.serial_Mk(vidx)
    compute_all_Mk(exp, device=device_str)

    t_ch = time_chinook_serial_mk(exp, vidx)
    t_gr = time_grizzly_mk(exp, device_str)
    speedup = t_ch / t_gr if t_gr > 0 else float("inf")
    notes = f"valid peaks={n_states}"
    return (
        f"| matrix_elements | {grid_str} | {n_states} | "
        f"{t_ch:.4f} | {t_gr:.4f} | {speedup:.2f}x | {notes} |"
    )


def append_section_header(path: Path, device_str: str, title: str) -> None:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gpu = device_label() if device_str == "cuda" else device_str
    with path.open("a", encoding="utf-8") as f:
        f.write(
            f"\n## {title}\n\n"
            f"- when: {when}\n"
            f"- host: cuda-host\n"
            f"- grizzly device: {device_str} ({gpu})\n"
            f"- chinook: CPU `serial_Mk`\n"
            f"- python: {sys.version.split()[0]}\n"
            f"- method: 1 warmup + 3 timed runs, median\n\n"
            "| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |\n"
            "|-----------|------|--------|-----------|-----------|---------|-------|\n"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark chinook serial_Mk vs compute_all_Mk")
    p.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Device for Grizzly compute_all_Mk (chinook always CPU)",
    )
    p.add_argument(
        "--append",
        metavar="SECTION",
        help='Append rows under "## SECTION" instead of overwriting RESULTS.md',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    results_path = ROOT / "benchmarks" / "RESULTS.md"

    print(
        f"GrizzlyME bench_me (Task 6a) — device={device_str} "
        f"(auto={device_label()})"
    )

    if args.append:
        append_section_header(results_path, device_str, args.append)
    else:
        write_results_header(results_path)

    for label, (nx, ny, ne) in GRIDS.items():
        row = run_grid(label, nx, ny, ne, device_str)
        append_result_row(results_path, row)
        print(f"    {row}")

    print(f"\nWrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
