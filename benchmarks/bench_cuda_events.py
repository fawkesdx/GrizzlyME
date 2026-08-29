#!/usr/bin/env python3
"""Task 6c: CUDA Event timing for compute_all_Mk (skip if no CUDA)."""

from __future__ import annotations

import collections
import collections.abc
import sys
from datetime import datetime, timezone
from pathlib import Path

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks._bench_common import append_result_row, device_label, make_graphene_experiment  # noqa: E402
from grizzly.engine import compute_all_Mk  # noqa: E402


def append_section(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def write_skip(path: Path) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    section = (
        f"\n## GPU (Task 6c) — skipped (no CUDA)\n\n"
        f"- when: {when}\n"
        f"- cuda available: False\n"
        f"- auto device: {device_label()}\n"
        f"- reason: No NVIDIA CUDA GPU on this machine. "
        f"Apple MPS is available but GrizzlyME engine uses float64/complex128; "
        f"MPS rejects float64 (see Task 6a CPU-forced benchmarks).\n"
        f"- action: Run this script on a CUDA machine to append Event-timed row.\n"
        f"- command: `python benchmarks/bench_cuda_events.py`\n\n"
    )
    append_section(path, section)
    return section


def run_cuda_bench(path: Path, grid: tuple[int, int, int] = (10, 10, 10)) -> str:
    nx, ny, ne = grid
    device = "cuda"
    exp = make_graphene_experiment(nx, ny, ne)

    # warmup
    compute_all_Mk(exp, device=device)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    compute_all_Mk(exp, device=device)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)

    n_states = int((exp.th >= 0).sum())
    row = (
        f"| cuda_event_mk | {nx}x{ny}x{ne} | {n_states} | — | {ms/1000:.6f} | — | "
        f"torch.cuda.Event; compute_all_Mk only; GPU={torch.cuda.get_device_name(0)} |"
    )

    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    append_section(
        path,
        f"\n## GPU (Task 6c)\n\n"
        f"- when: {when}\n"
        f"- cuda: {torch.cuda.get_device_name(0)}\n"
        f"- method: torch.cuda.Event around one `compute_all_Mk` after warmup\n\n"
        "| benchmark | grid | states | chinook_s | grizzly_s | speedup | notes |\n"
        "|-----------|------|--------|-----------|-----------|---------|-------|\n",
    )
    append_result_row(path, row)
    return row


def main() -> int:
    results = ROOT / "benchmarks" / "RESULTS.md"
    if not results.exists():
        results.write_text("# GrizzlyME benchmarks\n\n", encoding="utf-8")

    if not torch.cuda.is_available():
        write_skip(results)
        print("Task 6c: skipped (no CUDA). Appended skip section to benchmarks/RESULTS.md")
        return 0

    row = run_cuda_bench(results)
    print(f"Task 6c: CUDA Event timing done.\n  {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
