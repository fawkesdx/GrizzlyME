#!/usr/bin/env python3
"""Generate publication figures from benchmarks/RESULTS.md data.

Run from GrizzlyME root:
    python benchmarks/plot_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "website" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Sourced from benchmarks/RESULTS.md (2026-08-28). Update after re-bench.
ME_BENCH = {
    "labels": ["5³\n(36)", "10³\n(108)", "15³\n(260)"],
    "states": [36, 108, 260],
    "cpu": {
        "chinook_s": [0.0022, 0.0061, 0.0155],
        "grizzly_s": [0.0005, 0.0005, 0.0007],
        "speedup": [4.19, 12.94, 22.08],
        "platform": "CPU",
    },
    "cuda": {
        "chinook_s": [0.0200, 0.0205, 0.0481],
        "grizzly_s": [0.0032, 0.0023, 0.0035],
        "speedup": [6.19, 8.92, 13.91],
        "platform": "CUDA GPU",
    },
}

SPECTRAL_BENCH = {
    "labels": ["5³", "10³"],
    "states": [36, 108],
    "speedup": [1.76, 4.19],
}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")


def fig_me_speedup():
    """Bar chart: speedup vs chinook serial_Mk by platform."""
    x = np.arange(len(ME_BENCH["states"]))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(
        x - w / 2,
        ME_BENCH["cpu"]["speedup"],
        width=w,
        label=ME_BENCH["cpu"]["platform"],
        color="#2563eb",
    )
    ax.bar(
        x + w / 2,
        ME_BENCH["cuda"]["speedup"],
        width=w,
        label=ME_BENCH["cuda"]["platform"],
        color="#059669",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ME_BENCH["labels"])
    ax.set_xlabel("Datacube grid (valid peaks)")
    ax.set_ylabel("Speedup vs chinook serial_Mk")
    ax.set_title("Matrix-element batch speedup (GrizzlyME compute_all_Mk)")
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_me_speedup.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_me_wallclock():
    """Grouped bars: absolute ME wall time by platform."""
    x = np.arange(len(ME_BENCH["states"]))
    w = 0.2
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    series = [
        ("chinook CPU", ME_BENCH["cpu"]["chinook_s"], "#94a3b8"),
        ("Grizzly CPU", ME_BENCH["cpu"]["grizzly_s"], "#2563eb"),
        ("chinook CPU (remote)", ME_BENCH["cuda"]["chinook_s"], "#cbd5e1"),
        ("Grizzly CUDA", ME_BENCH["cuda"]["grizzly_s"], "#059669"),
    ]
    for i, (label, vals, color) in enumerate(series):
        ax.bar(x + offsets[i], vals, width=w, label=label, color=color)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ME_BENCH["labels"])
    ax.set_xlabel("Datacube grid (valid peaks)")
    ax.set_ylabel("Wall time (s, log scale)")
    ax.set_title("Matrix-element wall time (median of 3 runs)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_me_wallclock.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_me_speedup_vs_states():
    """Line plot: speedup vs number of valid peaks."""
    fig, ax = plt.subplots(figsize=(6, 4))
    states = ME_BENCH["states"]
    ax.plot(
        states,
        ME_BENCH["cpu"]["speedup"],
        "o-",
        color="#2563eb",
        linewidth=2,
        markersize=8,
        label=ME_BENCH["cpu"]["platform"],
    )
    ax.plot(
        states,
        ME_BENCH["cuda"]["speedup"],
        "s-",
        color="#059669",
        linewidth=2,
        markersize=8,
        label=ME_BENCH["cuda"]["platform"],
    )
    ax.set_xlabel("Valid peaks (states)")
    ax.set_ylabel("Speedup vs chinook serial_Mk")
    ax.set_title("ME speedup scales with batch size")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_me_speedup_vs_states.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_spectral_speedup():
    """Spectral assembly speedup (shared Mk, Task 6b)."""
    x = np.arange(len(SPECTRAL_BENCH["states"]))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, SPECTRAL_BENCH["speedup"], color="#7c3aed", width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{l}\n({s})" for l, s in zip(SPECTRAL_BENCH["labels"], SPECTRAL_BENCH["states"])]
    )
    ax.set_xlabel("Datacube grid (valid peaks)")
    ax.set_ylabel("Speedup vs chinook spectral()")
    ax.set_title("Spectral assembly (identical Mk)")
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_spectral_speedup.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_manifest():
    lines = [
        "# GrizzlyME publication figures",
        "",
        "Generated by `benchmarks/plot_figures.py` from `benchmarks/RESULTS.md`.",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `fig_me_speedup.png/pdf` | ME speedup CPU vs CUDA GPU |",
        "| `fig_me_wallclock.png/pdf` | Absolute ME times (log scale) |",
        "| `fig_me_speedup_vs_states.png/pdf` | Speedup vs valid peak count |",
        "| `fig_spectral_speedup.png/pdf` | Spectral-only speedup (shared Mk) |",
        "",
        "Regenerate: `python benchmarks/plot_figures.py`",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 100,
        }
    )
    fig_me_speedup()
    fig_me_wallclock()
    fig_me_speedup_vs_states()
    fig_spectral_speedup()
    write_manifest()
    print(f"Wrote figures to {OUT}/")
    for p in sorted(OUT.glob("fig_*")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
