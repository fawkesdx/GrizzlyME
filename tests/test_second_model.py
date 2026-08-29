"""Second-model end-to-end parity: Si-like s/p diatomic (not graphene)."""

from __future__ import annotations

import collections
import collections.abc

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import numpy as np
import pytest

import chinook.ARPES_lib as arpes_lib
import chinook.build_lib as build_lib

from grizzly import GrizzlyExperiment


def build_si_like_tb_arpes(n_kx: int = 8, n_ky: int = 8, n_e: int = 10):
    """Two-site Si-like TB: 8 orbitals (s, px, py, pz × 2), spinless.

    Same construction as ``tests/test_engine.create_multi_orbital_experiment``
    but returns TB + ARPES_dict for GrizzlyExperiment / chinook compare.
    """
    basis_dict = build_lib.gen_basis(
        {
            "atoms": [0, 1],
            "Z": {0: 14, 1: 14},
            "pos": [np.array([0.0, 0.0, 0.0]), np.array([1.35, 1.35, 1.35])],
            "orbs": [["30", "31x", "31y", "31z"], ["30", "31x", "31y", "31z"]],
            "spin": {"bool": False},
        }
    )
    a_lattice = 5.43 * np.eye(3)
    hopping_list = []
    for i in range(4):
        e_on = -5.0 if i == 0 else 0.0
        hopping_list.append([i, i, 0.0, 0.0, 0.0, e_on])
        hopping_list.append([i + 4, i + 4, 0.0, 0.0, 0.0, e_on])
    hopping_list.append([0, 4, 1.35, 1.35, 1.35, -2.0])
    hopping_list.append([1, 5, 1.35, 1.35, 1.35, 1.5])
    hopping_list.append([2, 6, 1.35, 1.35, 1.35, 1.5])
    hopping_list.append([3, 7, 1.35, 1.35, 1.35, 1.5])

    tb = build_lib.gen_TB(
        basis_dict,
        {
            "type": "list",
            "list": hopping_list,
            "a": a_lattice.tolist(),
            "cutoff": 6.0,
            "spin": {"bool": False},
        },
    )
    arpes_dict = {
        "cube": {
            "Tx": (-10.0, 10.0, n_kx),
            "Ty": (-10.0, 10.0, n_ky),
            "E": (-6.0, 2.0, n_e),
        },
        "hv": 30.0,
        "W": 4.5,
        "pol": np.array([0.0, 1.0, 0.0]),
        "T": 100.0,
        "resolution": {"E": 0.05, "k": 0.02},
        "SE": ["constant", 0.1],
    }
    return tb, arpes_dict


def _rel_l2(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-15))


def test_si_like_mk_vs_chinook_same_eigenpairs():
    """Mk: Grizzly compute_all_Mk vs chinook serial_Mk with chinook diagonalize.

    Uses ``use_grizzly_diagonalize=False`` so peak list / Ev match chinook.
    """
    tb, arpes_dict = build_si_like_tb_arpes()

    exp = arpes_lib.experiment(tb, arpes_dict)
    assert exp.datacube() is True
    mk_c = np.copy(exp.Mk)

    g = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    assert g.datacube(
        skip_chinook_mk=True,
        use_grizzly_diagonalize=False,
    ) is True

    assert g.Mk.shape == mk_c.shape
    max_diff = float(np.max(np.abs(g.Mk - mk_c)))
    assert max_diff < 1e-6, f"Si-like Mk max |Δ|={max_diff:.2e}"


def test_si_like_spectral_shared_mk():
    """Spectral assembly with identical Mk/pks (chinook setup + Grizzly spectral)."""
    tb, arpes_dict = build_si_like_tb_arpes()

    exp = arpes_lib.experiment(tb, arpes_dict)
    assert exp.datacube() is True

    # Grizzly uses same eigenpairs / peaks as chinook, then swaps Mk for shared compare
    g = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    assert g.datacube(
        skip_chinook_mk=True,
        use_grizzly_diagonalize=False,
    ) is True
    g._exp.Mk = exp.Mk.copy()

    I_c, Ig_c = exp.spectral()[:2]
    I_g, Ig_g = g.spectral()[:2]

    rel_I = _rel_l2(I_c, I_g)
    rel_Ig = _rel_l2(Ig_c, Ig_g)
    assert rel_I < 1e-4, f"shared-Mk raw I rel L2={rel_I:.2e}"
    assert rel_Ig < 0.05, f"shared-Mk Ig rel L2={rel_Ig:.2e}"


def test_si_like_full_pipeline_grizzly_me_spectral():
    """Full Grizzly path (chinook diag, Grizzly ME+spectral) vs chinook spectral."""
    tb, arpes_dict = build_si_like_tb_arpes()

    exp = arpes_lib.experiment(tb, arpes_dict)
    assert exp.datacube() is True
    I_c, Ig_c = exp.spectral()[:2]

    g = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    assert g.datacube(
        skip_chinook_mk=True,
        use_grizzly_diagonalize=False,
    ) is True
    I_g, Ig_g = g.spectral()[:2]

    rel_I = _rel_l2(I_c, I_g)
    rel_Ig = _rel_l2(Ig_c, Ig_g)
    assert rel_I < 1e-4, f"full ME+spectral raw I rel L2={rel_I:.2e}"
    assert rel_Ig < 0.05, f"full ME+spectral Ig rel L2={rel_Ig:.2e}"
