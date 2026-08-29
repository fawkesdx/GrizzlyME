"""Phase 4 Task 1: Fermi-Dirac distribution (chinook parity)."""

import collections
import collections.abc
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import numpy as np
import pytest
import torch

from chinook.ARPES_lib import kb, q, vf


def _chinook_fermi_ref(omega, T, mu=0.0):
    """Reference: chinook experiment.T_distribution + con_ferm."""
    ekbt = (omega - mu) / (kb * T / q)
    return np.asarray(vf(ekbt), dtype=np.float64)


def test_fermi():
    from grizzly.spectral import compute_fermi_distribution

    T = 300.0
    mu = 0.0

    # energies far below / above Fermi; include overflow case
    omega = torch.tensor(
        [-10.0, -1.0, 0.0, 1.0, 10.0, 100.0],
        dtype=torch.float64,
    )

    result = compute_fermi_distribution(omega, T)

    assert result.dtype == torch.float64
    assert result.shape == omega.shape

    ref = _chinook_fermi_ref(omega.numpy(), T, mu)
    torch.testing.assert_close(result, torch.from_numpy(ref), atol=1e-15, rtol=1e-14)

    assert (result >= 0).all()
    assert (result <= 1).all()
    assert result[0] > 0.99
    assert result[-1] == 0.0  # overflow guard (ekbt >> 709)


def test_m_factor():
    from grizzly.spectral import compute_M_factor

    rng = np.random.default_rng(42)
    B = 5
    Mk = torch.from_numpy(
        rng.standard_normal((B, 2, 3)) + 1j * rng.standard_normal((B, 2, 3))
    ).to(dtype=torch.complex128)
    pol_sph = torch.from_numpy(
        rng.standard_normal(3) + 1j * rng.standard_normal(3)
    ).to(dtype=torch.complex128)

    result = compute_M_factor(Mk, pol_sph)

    assert result.dtype == torch.float64
    assert result.shape == (B,)

    # chinook spinless momentum formula
    ref = np.sum(
        np.abs(np.einsum("ijk,k->ij", Mk.numpy(), pol_sph.numpy())) ** 2,
        axis=1,
    )
    torch.testing.assert_close(result, torch.from_numpy(ref), atol=1e-12, rtol=0)


def _spectral_intensity_loop(M_factor, pks, omega, SE, fermi, grid_shape):
    """Reference peak loop (chinook formula)."""
    Ny, Nx, Nw = grid_shape
    I = np.zeros((Ny, Nx, Nw), dtype=np.float64)
    omega_np = omega.detach().cpu().numpy()
    fermi_np = fermi.detach().cpu().numpy()
    M_np = M_factor.detach().cpu().numpy()
    pks_np = pks.detach().cpu().numpy()

    se_is_scalar = isinstance(SE, (complex, float, int, np.complexfloating, np.floating))

    for p in range(M_np.shape[0]):
        ky = int(np.real(pks_np[p, 1]))
        kx = int(np.real(pks_np[p, 2]))
        E_p = float(np.real(pks_np[p, 3]))
        for w_idx, w in enumerate(omega_np):
            if se_is_scalar:
                se_val = complex(SE) - 0.00005j
            else:
                se_val = complex(SE[w_idx]) - 0.00005j
            z = w - E_p - se_val
            I[ky, kx, w_idx] += M_np[p] * np.imag(-1.0 / (np.pi * z)) * fermi_np[w_idx]
    return I


def test_spectral_intensity():
    from grizzly.spectral import compute_spectral_intensity

    Ny, Nx, Nw = 4, 5, 8
    grid_shape = (Ny, Nx, Nw)
    omega = torch.linspace(-2.0, 2.0, Nw, dtype=torch.float64)
    fermi = torch.linspace(0.2, 0.9, Nw, dtype=torch.float64)
    SE = 0.05 + 0.02j  # constant broadening (complex scalar)

    # three peaks; two share a pixel to exercise accumulation
    pks = torch.tensor(
        [
            [0.0, 1.0, 2.0, -0.5],
            [0.0, 1.0, 2.0, 0.3],
            [0.0, 3.0, 4.0, 0.8],
        ],
        dtype=torch.float64,
    )
    M_factor = torch.tensor([1.5, 0.7, 2.2], dtype=torch.float64)

    result = compute_spectral_intensity(M_factor, pks, omega, SE, fermi, grid_shape)

    assert result.dtype == torch.float64
    assert result.shape == grid_shape

    ref = _spectral_intensity_loop(M_factor, pks, omega, SE, fermi, grid_shape)
    torch.testing.assert_close(result, torch.from_numpy(ref), atol=1e-10, rtol=0)

    # non-zero only at peak locations
    assert result[1, 2].abs().sum() > 0
    assert result[3, 4].abs().sum() > 0
    assert result[0, 0].abs().sum() == 0


def test_gaussian_fft_vs_scipy():
    from grizzly.spectral import gaussian_convolution_3d
    from scipy import ndimage

    shape = (31, 27, 23)
    I_np = np.zeros(shape, dtype=np.float64)
    I_np[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1.0
    sig = (2.5, 1.8, 3.2)

    ref = ndimage.gaussian_filter(I_np, sigma=sig)
    got = gaussian_convolution_3d(torch.from_numpy(I_np), *sig).numpy()

    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-6)


def test_chinook_raw_I_smoke():
    from grizzly.spectral import (
        build_raw_I_from_experiment,
        chinook_gaussian_sigmas,
        gaussian_convolution_3d,
    )

    from tests.test_engine import create_graphene_experiment

    exp = create_graphene_experiment()
    I_ref, Ig_ref = exp.spectral()

    I = build_raw_I_from_experiment(exp)
    I_ref_t = torch.from_numpy(I_ref)
    rel_I = torch.linalg.norm(I - I_ref_t) / torch.linalg.norm(I_ref_t)
    assert rel_I < 1e-4, f"raw I rel L2 {rel_I.item():.2e}"

    # Ig: FFT uses circular conv; chinook/scipy use reflect on small cube (~5.7e-3).
    # Optional check only — raw I is the gate for Task 4.
    kyg, kxg, wg = chinook_gaussian_sigmas(exp)
    Ig = gaussian_convolution_3d(I, kyg, kxg, wg)
    Ig_ref_t = torch.from_numpy(Ig_ref)
    rel_Ig = torch.linalg.norm(Ig - Ig_ref_t) / torch.linalg.norm(Ig_ref_t)
    assert rel_Ig < 0.01  # sanity only; see RESULTS.md
