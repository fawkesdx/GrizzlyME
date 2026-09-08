"""Reusable basis-level setup for Chinook matrix elements."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch

from chinook.ARPES_lib import all_Y, projection_map
import chinook.radint_lib as radint_lib

from .radint_cache import (
    DEFAULT_RADINT_CACHE,
    dig_range_from_cube,
    make_radint_cache_key,
    with_backend_tag,
)
from .radint_torch import make_radint_pointer_torch
from .ylm_torch import batched_gaunt

RADINT_MODES = ("auto", "torch", "chinook")
YLM_MODES = ("auto", "torch", "chinook")


@dataclass
class PreparedMeShell:
    """Basis-level matrix-element tables reusable across datacubes."""

    basis: Any
    prefactors: np.ndarray
    Largs: Any
    Margs: Any
    Gbasis: Any
    proj_arr: Any
    Bfuncs: Any
    radint_pointers: Any
    nstates: int
    spin: bool
    mfp: float
    radint_source: str
    ylm_source: str = "chinook"


def resolve_radint_mode(radint_mode: Optional[str] = None) -> str:
    """Explicit argument wins, then ``$GRIZZLY_RADINT``, then ``'auto'``."""
    raw = radint_mode or os.environ.get("GRIZZLY_RADINT") or "auto"
    mode = str(raw).strip().lower()
    if mode not in RADINT_MODES:
        raise ValueError(
            f"radint_mode must be one of {RADINT_MODES}, got {mode!r}"
        )
    return mode


def resolve_ylm_mode(ylm_mode: Optional[str] = None) -> str:
    """Explicit argument wins, then ``$GRIZZLY_YLM``, then ``'auto'``."""
    raw = ylm_mode or os.environ.get("GRIZZLY_YLM") or "auto"
    mode = str(raw).strip().lower()
    if mode not in YLM_MODES:
        raise ValueError(
            f"ylm_mode must be one of {YLM_MODES}, got {mode!r}"
        )
    return mode


def all_Y_torch(basis, *, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chinook ``all_Y`` using ``batched_gaunt`` for the Gaunt tables.

    Layout matches ``chinook.ARPES_lib.all_Y``: unique ``(l,m)`` rows, each with
    final-state ``L/M`` grids of shape ``(2, 3)`` and Gaunt coeffs for
    ``dl ∈ {-1,+1}``, ``dm ∈ {-1,0,+1}``.
    """
    maxproj = max(len(o.proj) for o in basis)
    lm_inds: list[tuple[int, int]] = []
    l_args: list[np.ndarray] = []
    m_args: list[np.ndarray] = []
    g_arr: list[np.ndarray] = []
    orb_point: list[np.ndarray] = []

    for o in basis:
        point = np.zeros(maxproj)
        for pi in range(len(o.proj)):
            p = o.proj[pi]
            lm = (int(p[2]), int(p[3]))
            if lm not in lm_inds:
                l0, m0 = lm
                L = np.array([[l0 - 1] * 3, [l0 + 1] * 3], dtype=int)
                M = np.tile(np.array([m0 - 1, m0, m0 + 1], dtype=int), (2, 1))
                dls = (-1, -1, -1, 1, 1, 1)
                dms = (-1, 0, 1, -1, 0, 1)
                g = (
                    batched_gaunt(
                        [l0] * 6,
                        [m0] * 6,
                        dls,
                        dms,
                        device=device,
                        dtype=torch.float64,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(2, 3)
                )
                l_args.append(L)
                m_args.append(M)
                g_arr.append(g)
                lm_inds.append(lm)
            point[pi] = lm_inds.index(lm)
        orb_point.append(point)

    return (
        np.asarray(l_args),
        np.asarray(m_args),
        np.asarray(g_arr),
        np.asarray(orb_point).astype(int),
    )


def prepare_me_shell(
    exp,
    *,
    radint_mode: Optional[str] = None,
    ylm_mode: Optional[str] = None,
    device: Optional[str] = None,
) -> PreparedMeShell:
    """Prepare Chinook radial-integral and spherical-harmonic tables.

    Basis rotation occurs here, matching Chinook's matrix-element setup.

    ``radint_mode`` (or ``$GRIZZLY_RADINT``) selects the radial-integral
    backend: ``'auto'`` prefers the torch Slater path and falls back to
    chinook on any failure, ``'torch'`` and ``'chinook'`` force one backend.
    A forced backend never reuses a cache entry built by the other one.

    ``ylm_mode`` (or ``$GRIZZLY_YLM``) selects the ``all_Y`` / Gaunt setup
    backend the same way (default ``auto`` → torch, chinook fallback).
    """
    mode = resolve_radint_mode(radint_mode)
    ymode = resolve_ylm_mode(ylm_mode)
    torch_device = str(device or getattr(exp, "device", None) or "cpu")

    exp.basis = exp.rot_basis()
    # 'auto' accepts either backend, so it shares one entry; a forced mode gets
    # its own namespace or the other backend's build would satisfy it.
    key = with_backend_tag(
        make_radint_cache_key(exp), None if mode == "auto" else mode
    )
    dig_range = dig_range_from_cube(exp.cube)
    source = "cache"

    def _rad_dict():
        return {
            "hv": exp.hv,
            "W": exp.W,
            "rad_type": exp.rad_type,
            "rad_args": exp.rad_args,
            "phase_shifts": exp.phase_shifts,
        }

    def _build_chinook():
        nonlocal source
        source = "chinook"
        return radint_lib.make_radint_pointer(_rad_dict(), exp.basis, dig_range)

    def _build_torch():
        nonlocal source
        source = "torch"
        return make_radint_pointer_torch(
            _rad_dict(), exp.basis, dig_range, device=torch_device
        )

    def _build_radint():
        if mode == "chinook":
            return _build_chinook()
        if mode == "torch":
            return _build_torch()
        try:
            return _build_torch()
        except Exception as exc:
            warnings.warn(
                f"torch radial integrals unavailable ({exc}); "
                "falling back to chinook",
                RuntimeWarning,
                stacklevel=2,
            )
            return _build_chinook()

    Bfuncs, radint_pointers = DEFAULT_RADINT_CACHE.get_or_build(
        key, _build_radint
    )

    def _ylm_chinook():
        return all_Y(exp.basis), "chinook"

    def _ylm_torch():
        return all_Y_torch(exp.basis, device=torch_device), "torch"

    def _build_ylm():
        if ymode == "chinook":
            return _ylm_chinook()
        if ymode == "torch":
            return _ylm_torch()
        try:
            return _ylm_torch()
        except Exception as exc:
            warnings.warn(
                f"torch all_Y unavailable ({exc}); falling back to chinook",
                RuntimeWarning,
                stacklevel=2,
            )
            return _ylm_chinook()

    (Largs, Margs, Gmats, orbital_pointers), ylm_source = _build_ylm()

    return PreparedMeShell(
        basis=exp.basis,
        prefactors=np.array(
            [
                orbital.sigma
                * np.exp((-0.5 / abs(exp.mfp)) * abs(orbital.depth))
                for orbital in exp.basis
            ]
        ),
        Largs=Largs,
        Margs=Margs,
        Gbasis=Gmats[orbital_pointers],
        proj_arr=projection_map(exp.basis),
        Bfuncs=Bfuncs,
        radint_pointers=radint_pointers,
        nstates=len(exp.basis),
        spin=bool(getattr(exp, "spin", False)),
        mfp=float(exp.mfp),
        radint_source=source,
        ylm_source=ylm_source,
    )
