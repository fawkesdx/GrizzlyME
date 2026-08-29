"""
Phase 3: compare grizzly.hamiltonian against chinook TB_model.solve_H
and against NumPy eigh on the same H(k).
"""

import collections
import collections.abc
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import numpy as np
import pytest
import torch

import chinook.build_lib as build_lib
import chinook.klib as klib

from grizzly.hamiltonian import (
    extract_hopping_data,
    build_and_diagonalize,
    solve_H,
)


def _make_graphene_TB(nk=12):
    """2-orbital graphene TB + dense k-mesh (same style as test_engine)."""
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
    tb = build_lib.gen_TB(basis_dict, hamiltonian_dict)
    kx = np.linspace(-0.3, 0.3, nk)
    ky = np.linspace(-0.3, 0.3, nk)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kpts = np.column_stack([KX.ravel(), KY.ravel(), np.zeros(KX.size)])
    tb.Kobj = klib.kpath(kpts)  # ndarray → uses pts as kpts directly
    return tb


def _fix_evec_phases(ref, test, tol=1e-8):
    """Match eigenvector columns up to U(1) gauge; return max |overlap| error."""
    # ref, test: (Nk, Nb, Nb) with columns = eigenvectors
    overlaps = np.einsum("kbi,kbj->kij", np.conj(ref), test)
    # for each k, band j: best phase from diagonal of V^H W after aligning
    max_err = 0.0
    Nk, Nb, _ = ref.shape
    for k in range(Nk):
        for j in range(Nb):
            # overlap of column j
            o = np.vdot(ref[k, :, j], test[k, :, j])
            if abs(o) < tol:
                # try matching to nearest column (degeneracy)
                o_row = overlaps[k, :, j]
                j2 = int(np.argmax(np.abs(o_row)))
                o = o_row[j2]
                aligned = test[k, :, j] * np.exp(-1j * np.angle(o))
                err = np.linalg.norm(ref[k, :, j2] - aligned)
            else:
                aligned = test[k, :, j] * np.exp(-1j * np.angle(o))
                err = np.linalg.norm(ref[k, :, j] - aligned)
            max_err = max(max_err, err)
    return max_err


def test_extract_hopping_data_shapes():
    tb = _make_graphene_TB(nk=4)
    data = extract_hopping_data(tb, device="cpu")
    assert data["i_indices"].ndim == 1
    assert data["j_indices"].shape == data["i_indices"].shape
    assert data["R_vectors"].shape == (data["i_indices"].shape[0], 3)
    assert data["amplitudes"].shape == data["i_indices"].shape
    assert data["n_basis"] == len(tb.basis)
    assert data["i_indices"].numel() > 0


def test_eigenvalues_match_chinook():
    tb = _make_graphene_TB(nk=8)
    E_ref, V_ref = tb.solve_H()
    E_g, V_g = solve_H(tb, device="cpu", chunk_size=0)
    E_g = E_g.detach().cpu().numpy()
    V_g = V_g.detach().cpu().numpy()
    assert E_ref.shape == E_g.shape
    np.testing.assert_allclose(E_g, E_ref, atol=1e-10, rtol=0)


def test_eigenvalues_match_numpy_eigh_uplo_u():
    """Same H(k) upper-triangle fill → NumPy eigh UPLO=U vs torch."""
    tb = _make_graphene_TB(nk=6)
    kpts = tb.Kobj.kpts
    Nb = len(tb.basis)
    Hmat = np.zeros((len(kpts), Nb, Nb), dtype=complex)
    for me in tb.mat_els:
        Hmat[:, me.i, me.j] = me.H2Hk()(kpts)
    E_np, _ = np.linalg.eigh(Hmat, UPLO="U")

    data = extract_hopping_data(tb, device="cpu")
    E_t, _ = build_and_diagonalize(kpts, data, device="cpu", chunk_size=0)
    np.testing.assert_allclose(E_t.detach().cpu().numpy(), E_np, atol=1e-10, rtol=0)


def test_eigenvectors_gauge_match_chinook():
    tb = _make_graphene_TB(nk=6)
    E_ref, V_ref = tb.solve_H()
    _, V_g = solve_H(tb, device="cpu")
    V_g = V_g.detach().cpu().numpy()
    err = _fix_evec_phases(V_ref, V_g)
    assert err < 1e-8, f"eigenvector gauge-fixed error {err}"


def test_chunked_matches_full():
    tb = _make_graphene_TB(nk=10)
    E_full, V_full = solve_H(tb, device="cpu", chunk_size=0)
    E_chunk, V_chunk = solve_H(tb, device="cpu", chunk_size=17)
    np.testing.assert_allclose(
        E_chunk.detach().cpu().numpy(),
        E_full.detach().cpu().numpy(),
        atol=1e-12,
        rtol=0,
    )
    err = _fix_evec_phases(
        V_full.detach().cpu().numpy(), V_chunk.detach().cpu().numpy()
    )
    assert err < 1e-10
