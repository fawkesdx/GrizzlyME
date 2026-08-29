"""GrizzlyExperiment — chinook ARPES experiment wrapper with Grizzly ME engine."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

import chinook.ARPES_lib as arpes_lib

import chinook.radint_lib as radint_lib

from grizzly.engine import compute_all_Mk
from grizzly.future import require_spinless
from grizzly.hamiltonian import solve_H
from grizzly.radint_cache import DEFAULT_RADINT_CACHE, make_radint_cache_key
from grizzly.utils import get_device, to_numpy


class GrizzlyExperiment:
    """Wrap chinook ``ARPES_lib.experiment``; route Mk through ``grizzly.engine``."""

    def __init__(self, TB, ARPES_dict, device: str = "auto"):
        self.TB = TB
        self.ARPES_dict = ARPES_dict
        self.device = get_device(device)
        self._exp = arpes_lib.experiment(TB, ARPES_dict)
        require_spinless(self._exp, feature="GrizzlyExperiment")
        # Process-wide by default so a new experiment with same TB/hv reuses radint.
        self._radint_cache = DEFAULT_RADINT_CACHE

    def __getattr__(self, name: str) -> Any:
        return getattr(self._exp, name)

    def datacube(
        self,
        ARPES_dict: Optional[dict] = None,
        diagonalize: bool = False,
        skip_chinook_mk: bool = True,
        use_grizzly_diagonalize: bool = True,
    ):
        """Run chinook ``datacube`` setup; matrix elements via Grizzly when skipping chinook ME.

        Parameters
        ----------
        skip_chinook_mk : bool
            If True (default), skip chinook ``serial_Mk`` / ``thread_Mk`` and fill
            ``Mk`` with ``compute_all_Mk`` once. Avoids paying for serial ME twice.
        use_grizzly_diagonalize : bool
            If True (default), patch ``TB.solve_H`` during datacube so the k-mesh
            diagonalization uses ``grizzly.hamiltonian.solve_H`` instead of chinook
            NumPy. Eigenpairs feed peak construction inside datacube (must run
            before ``pks`` are built, not after).

        Notes
        -----
        Slater ``make_radint_pointer`` results are cached in-process and on disk
        (``DEFAULT_RADINT_CACHE`` → ``~/.cache/grizzlyme/radint`` or
        ``$GRIZZLY_RADINT_CACHE_DIR``). Keyed by basis + hv/W/rad_type + energy window.
        Repeat ``datacube`` / new process with same key reuses radint.
        Disable disk with ``GRIZZLY_RADINT_DISK=0``. First cube of a brand-new
        model still pays full chinook radint cost once.
        """
        require_spinless(self._exp, feature="datacube")
        exp = self._exp
        tb = self.TB
        device_str = str(self.device)

        if ARPES_dict is not None:
            exp.update_pars(ARPES_dict, True)

        cache_key = make_radint_cache_key(exp)
        cached_radint = self._radint_cache.get(cache_key)

        orig_solve_h: Optional[Callable] = None
        if use_grizzly_diagonalize:

            def _grizzly_solve_h(Eonly: bool = False):
                Eband, Evec = solve_H(tb, device=device_str, Eonly=Eonly)
                tb.Eband = to_numpy(Eband)
                if Eonly or Evec is None:
                    tb.Evec = np.array([0])
                else:
                    tb.Evec = to_numpy(Evec)
                return tb.Eband, tb.Evec

            orig_solve_h = tb.solve_H
            tb.solve_H = _grizzly_solve_h

        orig_make_radint = radint_lib.make_radint_pointer
        if cached_radint is not None:
            Bfuncs_c, pointers_c = cached_radint

            def _cached_make_radint(_rad_dict, _basis, _Eb):
                return Bfuncs_c, pointers_c

            radint_lib.make_radint_pointer = _cached_make_radint

        try:
            if skip_chinook_mk:
                orig_serial = exp.serial_Mk
                orig_thread = exp.thread_Mk

                def _noop_mk(*_args, **_kwargs):
                    return None

                exp.serial_Mk = _noop_mk
                exp.thread_Mk = _noop_mk
                try:
                    result = exp.datacube(ARPES_dict=None, diagonalize=diagonalize)
                finally:
                    exp.serial_Mk = orig_serial
                    exp.thread_Mk = orig_thread
                self.ensure_Mk(force=True)
            else:
                result = exp.datacube(ARPES_dict=None, diagonalize=diagonalize)
        finally:
            radint_lib.make_radint_pointer = orig_make_radint
            if orig_solve_h is not None:
                tb.solve_H = orig_solve_h

        if cached_radint is None and hasattr(exp, "Bfuncs") and hasattr(
            exp, "radint_pointers"
        ):
            self._radint_cache.put(cache_key, exp.Bfuncs, exp.radint_pointers)

        return result

    def ensure_Mk(self, force: bool = False) -> np.ndarray:
        require_spinless(self._exp, feature="ensure_Mk")
        if force or not hasattr(self._exp, "Mk") or self._exp.Mk is None:
            self._exp.Mk = compute_all_Mk(self._exp, device=str(self.device))
        return self._exp.Mk

    @property
    def Mk(self) -> np.ndarray:
        if not hasattr(self._exp, "Mk") or self._exp.Mk is None:
            self.ensure_Mk()
        return self._exp.Mk

    def spectral(
        self,
        ARPES_dict: Optional[dict] = None,
        slice_select=None,
        add_map: bool = False,
        plot_bands: bool = False,
        ax=None,
        colourmap=None,
    ):
        """Build raw and broadened intensity maps via ``grizzly.spectral``."""
        from grizzly.spectral import (
            build_raw_I_from_experiment,
            chinook_gaussian_sigmas,
            gaussian_convolution_3d,
        )

        require_spinless(self._exp, feature="spectral")

        if ARPES_dict is not None and hasattr(self._exp, "update_pars"):
            self._exp.update_pars(ARPES_dict)

        self.ensure_Mk()

        I = build_raw_I_from_experiment(self._exp)
        kyg, kxg, wg = chinook_gaussian_sigmas(self._exp)
        Ig = gaussian_convolution_3d(I, kyg, kxg, wg)

        I_np = I.detach().cpu().numpy()
        Ig_np = Ig.detach().cpu().numpy()

        self._exp.I = I_np
        self._exp.Ig = Ig_np

        if slice_select is not None:
            return I_np, Ig_np, None
        return I_np, Ig_np
