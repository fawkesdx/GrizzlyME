"""Torch port of chinook's Slater-orbital radial integrals.

Mirrors ``chinook.radint_lib.make_radint_pointer`` for ``rad_type='slater'``
(the default when no ``rad_type`` is supplied). Every other radial-wavefunction
type raises :class:`NotImplementedError` so callers can fall back to chinook.

The energy x radius grid is evaluated as a batched tensor contraction instead
of chinook's per-energy adaptive recursion. The integration domain, the
spherical-Bessel order and the cubic interpolation over the five coarse energy
nodes all follow chinook exactly, so the returned ``B_array`` / ``B_pointers``
are drop-in replacements.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
from scipy.interpolate import interp1d

import chinook.electron_configs as econ

# Physical constants, identical to chinook.radint_lib.
hb = 6.626 * 10 ** -34 / (2 * np.pi)
q = 1.602 * 10 ** -19
A = 10.0 ** -10
me = 9.11 * 10 ** -31

# chinook.radint_lib.fill_radint_dic uses 5 coarse energies + cubic interp.
_N_COARSE = 5
# chinook.radint_lib.radint_calc integrates from 0.001 Angstrom.
_R_MIN = 0.001
# chinook.radint_lib.find_cutoff probe grid.
_CUTOFF_GRID = np.linspace(0, 25, 1000)[1:]
# Gauss-Legendre order; the integrand is smooth so this is machine-accurate.
_N_QUAD = 2048
# Below this argument the ascending series for j_l beats the closed form.
_JN_SERIES_MAX_X = 10.0
_JN_SERIES_TERMS = 60

_LEGGAUSS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n not in _LEGGAUSS_CACHE:
        _LEGGAUSS_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _LEGGAUSS_CACHE[n]


def spherical_jn_torch(l: int, x: torch.Tensor) -> torch.Tensor:
    """Spherical Bessel function of the first kind, ``j_l(x)``, for ``x >= 0``.

    The ascending series is used for small arguments where the closed form
    suffers catastrophic cancellation, and upward recurrence from ``j_0``,
    ``j_1`` elsewhere (stable once ``x > l``).
    """
    if l < 0:
        raise ValueError(f"spherical_jn_torch requires l >= 0, got {l}")

    small = x < _JN_SERIES_MAX_X

    x_small = torch.where(small, x, torch.ones_like(x))
    term = x_small ** l * (2.0 ** l * math.factorial(l) / math.factorial(2 * l + 1))
    series = term.clone()
    x2 = x_small * x_small
    for k in range(_JN_SERIES_TERMS):
        term = term * (
            -x2
            * (k + l + 1)
            / ((k + 1) * (2 * k + 2 * l + 3) * (2 * k + 2 * l + 2))
        )
        series = series + term

    x_large = torch.where(small, torch.ones_like(x), x)
    inv = 1.0 / x_large
    jm1 = torch.cos(x_large) * inv  # j_{-1}
    jl = torch.sin(x_large) * inv  # j_0
    for order in range(l):
        jm1, jl = jl, (2 * order + 1) * inv * jl - jm1

    return torch.where(small, series, jl)


def _slater_shape(Z: int, orb: str) -> tuple[float, float, float]:
    """``(ne, xi, amplitude)`` of chinook's ``electron_configs.Slater_exec``."""
    ne = econ.ndic[int(orb[0])]
    xi = econ.Z_eff(Z, orb) / ne
    amplitude = (2 * xi) ** ne * np.sqrt(2 * xi / math.gamma(float(2 * ne) + 1))
    return float(ne), float(xi), float(amplitude)


def _integrand(
    r: torch.Tensor,
    lp: int,
    k: torch.Tensor,
    ne: torch.Tensor,
    xi: torch.Tensor,
    amplitude: torch.Tensor,
) -> torch.Tensor:
    """Real part of chinook's ``general_Bnl_integrand`` for a Slater orbital.

    The ``(-1j)**lp`` prefactor is constant and applied to the finished
    integral instead, exactly as chinook applies its phase shift.
    """
    radial = amplitude * r ** (ne - 1.0) * torch.exp(-xi * r)
    return r ** 3 * spherical_jn_torch(lp, k * r) * radial


def _find_cutoff(magnitude: np.ndarray) -> float:
    """Port of ``chinook.radint_lib.find_cutoff`` for one integrand sample."""
    ftol = np.abs(magnitude).max() / 1e4
    sub_tol_inds = np.where(np.abs(magnitude) < ftol)[0]
    d_inds = np.diff(sub_tol_inds)
    if len(d_inds) == 0:
        raise NotImplementedError(
            "radial wavefunction is not localized within 25 Angstrom"
        )
    # First index whose suffix of gaps is all ones, i.e. contiguous tail.
    contiguous = d_inds == 1
    cutoff = len(d_inds)
    for i in range(len(d_inds) - 1, -1, -1):
        if not contiguous[i]:
            break
        cutoff = i
    if cutoff == len(d_inds):
        raise NotImplementedError(
            "radial wavefunction is not converged over 25 Angstrom"
        )
    return float(_CUTOFF_GRID[sub_tol_inds[cutoff]])


def _gen_orb_labels(basis) -> dict[str, list]:
    """Port of ``chinook.radint_lib.gen_orb_labels``."""
    orbitals: dict[str, list] = {}
    for o in basis:
        o_str = "{:d}-{:d}-{:d}".format(o.atom, o.n, o.l)
        if o_str not in orbitals:
            orbitals[o_str] = [o.Z, o.label]
    return orbitals


def _resolve_device(device) -> torch.device:
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "mps":
        # MPS has no float64; the radial integrals need it for chinook parity.
        return torch.device("cpu")
    return dev


def _coarse_energies(Eb) -> np.ndarray:
    if isinstance(Eb, (tuple, list, np.ndarray)):
        seq: Sequence = Eb
        if len(seq) < 2:
            raise NotImplementedError(f"unsupported energy range {Eb!r}")
        return np.linspace(float(seq[0]), float(seq[-1]), _N_COARSE)
    raise NotImplementedError(
        "torch radial integrals require an energy range (tuple of 2 floats)"
    )


def make_radint_pointer_torch(rad_dict: dict, basis, Eb, device="cpu"):
    """Torch equivalent of ``chinook.radint_lib.make_radint_pointer``.

    *args*:
        - **rad_dict**: ARPES parameter dictionary; keys ``'hv'``, ``'W'``,
          ``'phase_shifts'`` and optionally ``'rad_type'``.
        - **basis**: list of chinook orbital objects.
        - **Eb**: tuple of 2 floats, energy range in increasing order.

    *kwargs*:
        - **device**: torch device for the batched quadrature.

    *return*:
        - **B_array**: ``(N, 2)`` object array of executables of float.
        - **B_pointers**: integer array mapping basis order into ``B_array``.

    Raises :class:`NotImplementedError` for any non-Slater ``rad_type`` so the
    caller can fall back to chinook.
    """
    rad_type = rad_dict.get("rad_type") or "slater"
    if str(rad_type).lower() != "slater":
        raise NotImplementedError(
            f"torch radial integrals support 'slater' only, got {rad_type!r}"
        )

    hv = float(rad_dict["hv"])
    W = float(rad_dict.get("W", 0.0) or 0.0)
    phase_shifts = rad_dict.get("phase_shifts")

    energies = _coarse_energies(Eb)
    kinetic = (hv - W) + energies
    if np.any(kinetic < 0):
        raise NotImplementedError(
            "negative final-state kinetic energy in the requested range"
        )
    k_of_energy = np.sqrt(2.0 * me / hb ** 2 * kinetic * q) * A

    torch_device = _resolve_device(device)
    orbitals = _gen_orb_labels(basis)
    shapes = {o: _slater_shape(orbitals[o][0], orbitals[o][1]) for o in orbitals}

    # One job per (orbital shell, final l', coarse energy); batch them by l'
    # because the Bessel order must be a scalar.
    jobs_by_lp: dict[int, list[tuple[str, int]]] = {}
    for o in orbitals:
        l = int(o.split("-")[2])
        for lp in (l - 1, l + 1):
            if lp < 0:
                continue
            jobs_by_lp.setdefault(lp, []).extend((o, ei) for ei in range(_N_COARSE))

    integrals: dict[tuple[str, int], np.ndarray] = {}
    nodes, weights = _leggauss(_N_QUAD)
    nodes_t = torch.as_tensor(nodes, dtype=torch.float64, device=torch_device)
    weights_t = torch.as_tensor(weights, dtype=torch.float64, device=torch_device)
    grid_t = torch.as_tensor(
        _CUTOFF_GRID, dtype=torch.float64, device=torch_device
    )

    for lp, jobs in jobs_by_lp.items():
        k = torch.as_tensor(
            [k_of_energy[ei] for _, ei in jobs],
            dtype=torch.float64,
            device=torch_device,
        )[:, None]
        ne = torch.as_tensor(
            [shapes[o][0] for o, _ in jobs], dtype=torch.float64, device=torch_device
        )[:, None]
        xi = torch.as_tensor(
            [shapes[o][1] for o, _ in jobs], dtype=torch.float64, device=torch_device
        )[:, None]
        amplitude = torch.as_tensor(
            [shapes[o][2] for o, _ in jobs], dtype=torch.float64, device=torch_device
        )[:, None]

        probe = _integrand(grid_t[None, :], lp, k, ne, xi, amplitude)
        probe_np = probe.abs().cpu().numpy()
        upper = torch.as_tensor(
            [_find_cutoff(row) for row in probe_np],
            dtype=torch.float64,
            device=torch_device,
        )[:, None]

        half = 0.5 * (upper - _R_MIN)
        centre = 0.5 * (upper + _R_MIN)
        r = centre + half * nodes_t[None, :]
        values = _integrand(r, lp, k, ne, xi, amplitude)
        quad = (half * weights_t[None, :] * values).sum(dim=1).cpu().numpy()

        for (o, ei), value in zip(jobs, quad):
            integrals.setdefault(
                (o, lp), np.zeros(_N_COARSE, dtype=np.complex128)
            )[ei] = value

    B_dictionary = {}
    for o in orbitals:
        l = int(o.split("-")[2])
        for lp in (l - 1, l + 1):
            label = "{:s}-{:d}".format(o, lp)
            phase = 1.0 if phase_shifts is None else phase_shifts[label]
            if lp < 0:
                samples = np.zeros(_N_COARSE)
            else:
                samples = phase * ((-1.0j) ** lp) * integrals[(o, lp)]
            B_dictionary[label] = interp1d(energies, samples, kind="cubic")

    return _radint_dict_to_arr(B_dictionary, basis)


def _radint_dict_to_arr(Bdict: dict, basis):
    """Port of ``chinook.radint_lib.radint_dict_to_arr``."""
    Blist: list[list[Any]] = []
    lchan: list[str] = []
    pointers: list[int] = []
    for b in basis:
        orbstr = "{:d}-{:d}-{:d}".format(b.atom, b.n, b.l)
        if orbstr not in lchan:
            lchan.append(orbstr)
            Blist.append(
                [
                    Bdict[orbstr + "-{:d}".format(b.l - 1)],
                    Bdict[orbstr + "-{:d}".format(b.l + 1)],
                ]
            )
        pointers.append(lchan.index(orbstr))
    return np.array(Blist), np.array(pointers)
