"""Reusable basis-level setup for Chinook matrix elements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chinook.ARPES_lib import all_Y, projection_map
import chinook.radint_lib as radint_lib

from .radint_cache import (
    DEFAULT_RADINT_CACHE,
    dig_range_from_cube,
    make_radint_cache_key,
)


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


def prepare_me_shell(exp, *, radint_mode: str = "auto") -> PreparedMeShell:
    """Prepare Chinook radial-integral and spherical-harmonic tables.

    Basis rotation occurs here, matching Chinook's matrix-element setup.
    """
    if radint_mode == "torch":
        raise NotImplementedError("torch radial integrals require Phase A")

    exp.basis = exp.rot_basis()
    key = make_radint_cache_key(exp)
    dig_range = dig_range_from_cube(exp.cube)
    built = False

    def _build_radint():
        nonlocal built
        built = True
        rad_dict = {
            "hv": exp.hv,
            "W": exp.W,
            "rad_type": exp.rad_type,
            "rad_args": exp.rad_args,
            "phase_shifts": exp.phase_shifts,
        }
        return radint_lib.make_radint_pointer(rad_dict, exp.basis, dig_range)

    Bfuncs, radint_pointers = DEFAULT_RADINT_CACHE.get_or_build(
        key, _build_radint
    )
    Largs, Margs, Gmats, orbital_pointers = all_Y(exp.basis)

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
        radint_source="chinook" if built else "cache",
    )
