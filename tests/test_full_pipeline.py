"""End-to-end GrizzlyExperiment tests (Phase 5)."""

import collections
import collections.abc

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import numpy as np
import pytest

import chinook.ARPES_lib as arpes_lib
import chinook.build_lib as build_lib

from grizzly import GrizzlyExperiment


def build_graphene_tb_arpes():
    """Same 2-orbital graphene model as ``tests/test_engine.py``."""
    basis_args = {
        "atoms": [0, 1],
        "Z": {0: 6, 1: 6},
        "pos": [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.42, 0.0])],
        "orbs": [["21z"], ["21z"]],
        "spin": {"bool": False},
    }
    basis_dict = build_lib.gen_basis(basis_args)

    a_lattice = np.array(
        [
            [2.46, 0.0, 0.0],
            [-1.23, 2.130446, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )

    hopping_list = [
        [0, 0, 0.0, 0.0, 0.0, 0.0],
        [1, 1, 0.0, 0.0, 0.0, 0.0],
        [0, 1, 0.0, 1.42, 0.0, -2.8],
        [0, 1, 1.23, -0.71, 0.0, -2.8],
        [0, 1, -1.23, -0.71, 0.0, -2.8],
    ]

    hamiltonian_dict = {
        "type": "list",
        "list": hopping_list,
        "a": a_lattice.tolist(),
        "cutoff": 5.0,
        "spin": {"bool": False},
    }

    tb_model = build_lib.gen_TB(basis_dict, hamiltonian_dict)

    arpes_dict = {
        "cube": {
            "Tx": (-15.0, 15.0, 10),
            "Ty": (-15.0, 15.0, 10),
            "E": (-3.0, 3.0, 15),
        },
        "hv": 21.2,
        "W": 4.5,
        "pol": np.array([1.0, 0.0, 0.0]),
        "T": 300.0,
        "resolution": {"E": 0.05, "k": 0.02},
        "SE": ["constant", 0.1],
    }

    return tb_model, arpes_dict


def test_cpu_device():
    """GrizzlyExperiment with device='cpu': datacube + ensure_Mk succeed."""
    tb, arpes_dict = build_graphene_tb_arpes()
    g = GrizzlyExperiment(tb, arpes_dict, device="cpu")

    assert g.device.type == "cpu"
    assert g.datacube() is True

    mk = g.ensure_Mk()
    assert mk.shape[1:] == (2, 3)
    assert mk.shape[0] == len(g.pks)
    assert np.all(np.isfinite(mk.real))
    assert np.all(np.isfinite(mk.imag))


def test_datacube_mk_shape():
    """Construct GrizzlyExperiment, datacube, ensure Mk; check shape and finiteness."""
    tb, arpes_dict = build_graphene_tb_arpes()
    g = GrizzlyExperiment(tb, arpes_dict, device="cpu")

    assert g.datacube() is True

    mk = g.ensure_Mk()
    assert mk.shape[1:] == (2, 3)
    assert mk.shape[0] == len(g.pks)
    assert np.all(np.isfinite(mk.real))
    assert np.all(np.isfinite(mk.imag))

    assert g.Mk.shape == mk.shape
    assert np.any(np.abs(mk) > 0.0)


def test_datacube_skips_chinook_serial_mk():
    """Default datacube must not invoke chinook serial_Mk."""
    tb, arpes_dict = build_graphene_tb_arpes()
    g = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    calls = []

    def counting_serial(indices):
        calls.append(len(indices))

    g._exp.serial_Mk = counting_serial
    assert g.datacube() is True
    assert calls == []
    assert np.any(np.abs(g.Mk) > 0.0)


def test_datacube_mk_matches_chinook_serial():
    """Grizzly-only Mk path matches chinook serial_Mk on graphene."""
    tb, arpes_dict = build_graphene_tb_arpes()

    g_chinook = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    g_chinook.datacube(skip_chinook_mk=False)

    g_grizzly = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    g_grizzly.datacube(skip_chinook_mk=True)

    np.testing.assert_allclose(
        g_grizzly.Mk,
        g_chinook.Mk,
        rtol=1e-6,
        atol=1e-10,
    )


def test_datacube_grizzly_diagonalize_matches_chinook():
    """Grizzly solve_H inside datacube matches chinook Eb and Mk."""
    tb, arpes_dict = build_graphene_tb_arpes()

    g_np = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    g_np.datacube(use_grizzly_diagonalize=False, skip_chinook_mk=True)

    g_pt = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    g_pt.datacube(use_grizzly_diagonalize=True, skip_chinook_mk=True)

    np.testing.assert_allclose(g_pt._exp.Eb, g_np._exp.Eb, atol=1e-10)
    np.testing.assert_allclose(g_pt.Mk, g_np.Mk, rtol=1e-6, atol=1e-10)


def test_spinful_model_raises_v2_error():
    """Spinful experiment is rejected until GrizzlyME v2."""
    from grizzly import GrizzlyMEv2FeatureError

    tb, arpes_dict = build_graphene_tb_arpes()
    g = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    g._exp.spin = True
    with pytest.raises(GrizzlyMEv2FeatureError, match="GrizzlyME v2"):
        g.datacube()

def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-15))


def test_spectral_raw_I_vs_chinook():
    """Grizzly spectral raw I matches chinook with shared Mk (rel L2 < 1e-4)."""
    tb, arpes_dict = build_graphene_tb_arpes()

    exp = arpes_lib.experiment(tb, arpes_dict)
    assert exp.datacube() is True

    g = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    assert g.datacube() is True

    # Shared Mk: spectral-assembly-only compare (Task 5b).
    g._exp.Mk = exp.Mk.copy()

    I_c, Ig_c = exp.spectral()[:2]
    I_g, Ig_g = g.spectral()[:2]

    rel_I = _rel_l2(I_c, I_g)
    assert rel_I < 1e-4, f"shared-Mk raw I rel L2 {rel_I:.2e}"

    rel_Ig = _rel_l2(Ig_c, Ig_g)
    # FFT circular conv vs scipy reflect — sanity bound only (see test_spectral.py).
    assert rel_Ig < 0.01, f"shared-Mk Ig rel L2 {rel_Ig:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
