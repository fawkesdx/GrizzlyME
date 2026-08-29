#!/usr/bin/env python3
"""Publication parity figures: chinook vs GrizzlyME (graphene Dirac cut).

Run from GrizzlyME root:
    python benchmarks/plot_parity_figure.py
"""

from __future__ import annotations

import collections
import collections.abc
import sys
from pathlib import Path

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

import chinook.ARPES_lib as arpes_lib  # noqa: E402
import chinook.build_lib as build_lib  # noqa: E402
from grizzly import GrizzlyExperiment  # noqa: E402


def build_dirac_graphene():
    """1D kx cut through graphene Dirac point (same setup as plot_arpes_comparison.py)."""
    basis_dict = build_lib.gen_basis(
        {
            "atoms": [0, 1],
            "Z": {0: 6, 1: 6},
            "pos": [np.array([0.0, 0.0, 0.0]), np.array([0.5, np.sqrt(3) / 6, 0.0])],
            "orbs": [["21z"], ["21z"]],
            "spin": {"bool": False},
        }
    )
    a_lattice = np.array(
        [[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]]
    )
    dx1, dy1 = 0.5, np.sqrt(3) / 6
    dx2, dy2 = -0.5, np.sqrt(3) / 6
    dx3, dy3 = 0.0, -np.sqrt(3) / 3
    hopping_list = [
        [0, 1, dx1, dy1, 0.0, -2.8],
        [0, 1, dx2, dy2, 0.0, -2.8],
        [0, 1, dx3, dy3, 0.0, -2.8],
        [1, 0, -dx1, -dy1, 0.0, -2.8],
        [1, 0, -dx2, -dy2, 0.0, -2.8],
        [1, 0, -dx3, -dy3, 0.0, -2.8],
    ]
    tb_model = build_lib.gen_TB(
        basis_dict,
        {
            "type": "list",
            "list": hopping_list,
            "a": a_lattice.tolist(),
            "cutoff": 10.0,
            "spin": {"bool": False},
        },
    )
    arpes_dict = {
        "cube": {"X": (-4.0, 4.0, 300), "Y": (3.6276, 3.6276, 1), "E": (-10.0, 5.0, 300)},
        "hv": 100.0,
        "W": 4.5,
        "pol": np.array([1.0, 1.0, 1.0]),
        "T": 10.0,
        "resolution": {"E": 0.1, "k": 0.02},
        "SE": ["constant", 0.1],
    }
    return tb_model, arpes_dict


def rel_l2(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-15))


def run_parity(device: str = "cpu"):
    tb, arpes_dict = build_dirac_graphene()

    exp = arpes_lib.experiment(tb, arpes_dict)
    exp.datacube()
    exp.ang = 0.0
    _, Ig_c = exp.spectral()

    g = GrizzlyExperiment(tb, arpes_dict, device=device)
    g.datacube()
    g.ang = 0.0
    _, Ig_g = g.spectral()

    return Ig_c, Ig_g, exp.Mk, g.Mk


def fig_arpes_maps(Ig_c, Ig_g, extent_kx=(-4.0, 4.0), extent_e=(-10.0, 5.0)):
    """Three-panel: chinook, Grizzly, |difference|."""
    spec_c = Ig_c[0, :, :].T
    spec_g = Ig_g[0, :, :].T
    diff = np.abs(spec_c - spec_g)
    rel = rel_l2(spec_c, spec_g)

    vmax = max(np.max(spec_c), np.max(spec_g)) * 0.85
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    kw = dict(aspect="auto", origin="lower", cmap="magma", extent=[*extent_kx, *extent_e])

    axes[0].imshow(spec_c, vmax=vmax, **kw)
    axes[0].set_title("Chinook")
    axes[0].set_xlabel(r"$k_x$ (Å$^{-1}$)")
    axes[0].set_ylabel("Binding energy (eV)")

    axes[1].imshow(spec_g, vmax=vmax, **kw)
    axes[1].set_title("GrizzlyME")
    axes[1].set_xlabel(r"$k_x$ (Å$^{-1}$)")

    im = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="viridis", extent=[*extent_kx, *extent_e])
    axes[2].set_title(r"$|I_g - I_c|$  (rel L2 = {:.2e})".format(rel))
    axes[2].set_xlabel(r"$k_x$ (Å$^{-1}$)")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Graphene Dirac cut — broadened ARPES intensity parity", y=1.02, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_parity_arpes_map.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rel


def fig_scatter_parity(Ig_c, Ig_g):
    """1:1 scatter of flattened intensity values."""
    a = Ig_c.ravel()
    b = Ig_g.ravel()
    rel = rel_l2(a, b)
    max_diff = float(np.max(np.abs(a - b)))

    fig, ax = plt.subplots(figsize=(5, 5))
    mask = (a > 0) | (b > 0)
    ax.scatter(a[mask], b[mask], s=4, alpha=0.25, c="#2563eb", edgecolors="none")
    hi = max(a.max(), b.max()) * 1.05
    ax.plot([0, hi], [0, hi], "k--", linewidth=1, label="1:1")
    ax.set_xlabel("Chinook $I_g$")
    ax.set_ylabel("GrizzlyME $I_g$")
    ax.set_title("Spectral intensity parity")
    ax.text(
        0.05,
        0.95,
        f"rel L2 = {rel:.2e}\nmax |Δ| = {max_diff:.2e}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_parity_scatter.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rel, max_diff


def fig_mk_parity(Mk_c, Mk_g):
    """Complex Mk element scatter (real part vs real part)."""
    a = Mk_c.real.ravel()
    b = Mk_g.real.ravel()
    rel = rel_l2(a, b)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(a, b, s=8, alpha=0.5, c="#059669", edgecolors="none")
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
    ax.set_xlabel(r"Re($M_k$) chinook")
    ax.set_ylabel(r"Re($M_k$) GrizzlyME")
    ax.set_title("Matrix-element parity (real part)")
    ax.text(0.05, 0.95, f"rel L2 = {rel:.2e}", transform=ax.transAxes, va="top", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_parity_mk.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rel


def update_figures_readme(stats: dict):
    readme = OUT / "README.md"
    extra = [
        "",
        "| `fig_parity_arpes_map.png/pdf` | Chinook vs Grizzly Dirac-cut ARPES maps |",
        "| `fig_parity_scatter.png/pdf` | 1:1 scatter of broadened intensity |",
        "| `fig_parity_mk.png/pdf` | Matrix-element Re($M_k$) parity |",
        "",
        f"Parity stats: Ig rel L2 = {stats['Ig_rel_l2']:.2e}, Mk rel L2 = {stats['Mk_rel_l2']:.2e}",
        "",
        "Regenerate parity: `python benchmarks/plot_parity_figure.py`",
    ]
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if "fig_parity_arpes_map" not in text:
            readme.write_text(text.rstrip() + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    else:
        readme.write_text("# Figures\n" + "\n".join(extra) + "\n", encoding="utf-8")


def main() -> int:
    plt.rcParams.update({"font.size": 11, "figure.dpi": 100})
    print("Running chinook + GrizzlyME Dirac-cut parity ...")
    Ig_c, Ig_g, Mk_c, Mk_g = run_parity(device="cpu")

    ig_rel = fig_arpes_maps(Ig_c, Ig_g)
    scatter_rel, max_d = fig_scatter_parity(Ig_c, Ig_g)
    mk_rel = fig_mk_parity(Mk_c, Mk_g)

    stats = {"Ig_rel_l2": ig_rel, "Mk_rel_l2": mk_rel, "Ig_max_diff": max_d}
    update_figures_readme(stats)

    print(f"  Ig rel L2 = {ig_rel:.2e}  max |Δ| = {max_d:.2e}")
    print(f"  Mk rel L2 = {mk_rel:.2e}")
    print(f"Wrote parity figures to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
