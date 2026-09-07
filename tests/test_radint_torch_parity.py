"""Parity of the torch Slater radial integrals against chinook."""

from __future__ import annotations

import collections
import collections.abc

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import numpy as np
import pytest

import chinook.build_lib as build_lib
import chinook.radint_lib as radint_lib

from grizzly.radint_torch import make_radint_pointer_torch


def _tiny_slater_basis(orbs=None, Z=None, atoms=None):
    """Small multi-l basis built with the same helper as ``test_full_pipeline``."""
    basis_args = {
        "atoms": atoms if atoms is not None else [0, 1],
        "Z": Z if Z is not None else {0: 6, 1: 23},
        "pos": [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.42, 0.0])],
        "orbs": orbs if orbs is not None else [["20", "21z"], ["32xy", "32xz"]],
        "spin": {"bool": False},
    }
    return build_lib.gen_basis(basis_args)["bulk"]


def _rad_dict(rad_type=None, phase_shifts=None):
    d = {"hv": 21.2, "W": 4.5, "rad_args": None, "phase_shifts": phase_shifts}
    if rad_type is not None:
        d["rad_type"] = rad_type
    return d


def _assert_parity(basis, rad_dict, Eb, rtol=1e-5, atol=1e-6):
    B_c, P_c = radint_lib.make_radint_pointer(dict(rad_dict), basis, Eb)
    B_t, P_t = make_radint_pointer_torch(dict(rad_dict), basis, Eb, device="cpu")

    assert np.array_equal(P_c, P_t)
    assert B_c.shape == B_t.shape

    energies = np.linspace(Eb[0], Eb[1], 17)
    for i in range(len(B_c)):
        for j in (0, 1):
            yc = np.array([complex(B_c[i, j](float(e))) for e in energies])
            yt = np.array([complex(B_t[i, j](float(e))) for e in energies])
            assert np.allclose(yc, yt, rtol=rtol, atol=atol), (
                i,
                j,
                np.abs(yc - yt).max(),
            )


def test_slater_radint_matches_chinook_tiny_basis():
    basis = _tiny_slater_basis()
    _assert_parity(basis, _rad_dict("slater"), (-3.3, 3.3))


def test_missing_rad_type_defaults_to_slater():
    """No ``rad_type`` key must behave exactly like ``'slater'``."""
    basis = _tiny_slater_basis()
    _assert_parity(basis, _rad_dict(None), (-3.3, 3.3))


def test_graphene_pz_only_basis():
    """The 2-orbital graphene basis used across the pipeline tests."""
    basis = _tiny_slater_basis(orbs=[["21z"], ["21z"]], Z={0: 6, 1: 6})
    _assert_parity(basis, _rad_dict("slater"), (-3.6, 3.6))


def test_phase_shifts_are_applied():
    basis = _tiny_slater_basis(orbs=[["21z"], ["21z"]], Z={0: 6, 1: 6})
    phase = {}
    for orbital in basis:
        for lp in (orbital.l - 1, orbital.l + 1):
            phase[
                "{:d}-{:d}-{:d}-{:d}".format(
                    orbital.atom, orbital.n, orbital.l, lp
                )
            ] = 0.37
    _assert_parity(basis, _rad_dict("slater", phase_shifts=phase), (-3.6, 3.6))


def test_pointers_dedupe_shared_anl():
    """Orbitals sharing a-n-l share one row of ``B_array``."""
    basis = _tiny_slater_basis(orbs=[["21x", "21y", "21z"], ["21z"]], Z={0: 6, 1: 6})
    B_t, P_t = make_radint_pointer_torch(_rad_dict("slater"), basis, (-3.0, 3.0))
    assert list(P_t) == [0, 0, 0, 1]
    assert B_t.shape == (2, 2)


def test_hydrogenic_radint_matches_chinook_tiny_basis():
    basis = _tiny_slater_basis()
    _assert_parity(basis, _rad_dict("hydrogenic"), (-3.3, 3.3))


def test_hydrogenic_graphene_pz_only_basis():
    basis = _tiny_slater_basis(orbs=[["21z"], ["21z"]], Z={0: 6, 1: 6})
    _assert_parity(basis, _rad_dict("hydrogenic"), (-3.6, 3.6))


def test_hydrogenic_nodal_shells():
    """``n - l - 1 > 0`` shells exercise the generalized Laguerre polynomial."""
    basis = _tiny_slater_basis(orbs=[["30", "31z"], ["20"]], Z={0: 23, 1: 6})
    _assert_parity(basis, _rad_dict("hydrogenic"), (-3.3, 3.3))


def test_hydrogenic_rad_type_is_case_insensitive():
    basis = _tiny_slater_basis(orbs=[["21z"], ["21z"]], Z={0: 6, 1: 6})
    _assert_parity(basis, _rad_dict("Hydrogenic"), (-3.6, 3.6))


def test_hydrogenic_phase_shifts_are_applied():
    basis = _tiny_slater_basis(orbs=[["21z"], ["21z"]], Z={0: 6, 1: 6})
    phase = {}
    for orbital in basis:
        for lp in (orbital.l - 1, orbital.l + 1):
            phase[
                "{:d}-{:d}-{:d}-{:d}".format(
                    orbital.atom, orbital.n, orbital.l, lp
                )
            ] = 0.37
    _assert_parity(basis, _rad_dict("hydrogenic", phase_shifts=phase), (-3.6, 3.6))


def test_hydrogenic_differs_from_slater():
    """Guard against the hydrogenic branch silently reusing Slater shapes."""
    basis = _tiny_slater_basis(orbs=[["21z"], ["21z"]], Z={0: 6, 1: 6})
    B_s, _ = make_radint_pointer_torch(_rad_dict("slater"), basis, (-3.0, 3.0))
    B_h, _ = make_radint_pointer_torch(_rad_dict("hydrogenic"), basis, (-3.0, 3.0))
    assert not np.isclose(complex(B_s[0, 1](0.0)), complex(B_h[0, 1](0.0)))


@pytest.mark.parametrize("rad_type", ["grid", "exec", "fixed"])
def test_unsupported_rad_types_raise_not_implemented(rad_type):
    basis = _tiny_slater_basis(orbs=[["21z"], ["21z"]], Z={0: 6, 1: 6})
    with pytest.raises(NotImplementedError):
        make_radint_pointer_torch(_rad_dict(rad_type), basis, (-3.0, 3.0))


def test_torch_bfuncs_reproduce_pipeline_mk():
    """Swapping chinook Bfuncs for the torch ones leaves Mk unchanged."""
    import runpy
    from pathlib import Path

    from grizzly import GrizzlyExperiment, compute_all_Mk
    from grizzly.radint_cache import dig_range_from_cube

    helpers = runpy.run_path(str(Path(__file__).with_name("test_full_pipeline.py")))
    tb, arpes_dict = helpers["build_graphene_tb_arpes"]()

    g = GrizzlyExperiment(tb, arpes_dict, device="cpu")
    assert g.datacube() is True
    Mk_chinook = compute_all_Mk(g._exp, device="cpu")

    rad_dict = {
        "hv": g._exp.hv,
        "W": g._exp.W,
        "rad_type": g._exp.rad_type,
        "rad_args": g._exp.rad_args,
        "phase_shifts": g._exp.phase_shifts,
    }
    Bfuncs, pointers = make_radint_pointer_torch(
        rad_dict, g._exp.basis, dig_range_from_cube(g._exp.cube), device="cpu"
    )
    assert np.array_equal(pointers, g._exp.radint_pointers)

    g._exp.Bfuncs = Bfuncs
    Mk_torch = compute_all_Mk(g._exp, device="cpu")

    np.testing.assert_allclose(Mk_torch, Mk_chinook, rtol=1e-6, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
