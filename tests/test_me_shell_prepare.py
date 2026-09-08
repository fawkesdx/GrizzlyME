"""Tests for reusable matrix-element shell preparation."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path

import numpy as np
import pytest

from grizzly import GrizzlyExperiment, prepare_me_shell
from grizzly.radint_cache import RadintSetupCache


def _tiny_full_me_exp(rad_type=None):
    helpers = runpy.run_path(
        str(Path(__file__).with_name("test_full_pipeline.py"))
    )
    tb, arpes_dict = helpers["build_graphene_tb_arpes"]()
    if rad_type is not None:
        arpes_dict["rad_type"] = rad_type
    return GrizzlyExperiment(tb, arpes_dict, device="cpu")


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    import grizzly.me_shell as me_shell

    cache = RadintSetupCache(disk_dir=tmp_path, use_disk=False)
    monkeypatch.setattr(me_shell, "DEFAULT_RADINT_CACHE", cache)
    monkeypatch.delenv("GRIZZLY_RADINT", raising=False)
    monkeypatch.delenv("GRIZZLY_YLM", raising=False)
    return cache


def test_prepare_me_shell_second_call_is_cache(isolated_cache):
    exp = _tiny_full_me_exp()

    s1 = prepare_me_shell(exp)
    s2 = prepare_me_shell(exp)

    assert s1.radint_source == "torch"
    assert s2.radint_source == "cache"
    assert len(s1.radint_pointers) == len(s1.basis)


def test_chinook_mode_forces_chinook(isolated_cache):
    exp = _tiny_full_me_exp()
    shell = prepare_me_shell(exp, radint_mode="chinook")
    assert shell.radint_source == "chinook"


def test_torch_mode_forces_torch(isolated_cache):
    exp = _tiny_full_me_exp()
    shell = prepare_me_shell(exp, radint_mode="torch")
    assert shell.radint_source == "torch"


def test_env_var_selects_backend(isolated_cache, monkeypatch):
    monkeypatch.setenv("GRIZZLY_RADINT", "chinook")
    exp = _tiny_full_me_exp()
    assert prepare_me_shell(exp).radint_source == "chinook"


def test_explicit_mode_beats_env_var(isolated_cache, monkeypatch):
    monkeypatch.setenv("GRIZZLY_RADINT", "chinook")
    exp = _tiny_full_me_exp()
    assert prepare_me_shell(exp, radint_mode="torch").radint_source == "torch"


def test_auto_falls_back_to_chinook(isolated_cache, monkeypatch):
    """Unsupported rad_type must not break auto mode."""
    import grizzly.me_shell as me_shell

    def boom(*args, **kwargs):
        raise NotImplementedError("nope")

    monkeypatch.setattr(me_shell, "make_radint_pointer_torch", boom)
    exp = _tiny_full_me_exp()

    with pytest.warns(RuntimeWarning, match="falling back to chinook"):
        shell = prepare_me_shell(exp)
    assert shell.radint_source == "chinook"


def test_torch_mode_propagates_failure(isolated_cache, monkeypatch):
    """Explicit torch mode must not silently fall back."""
    import grizzly.me_shell as me_shell

    def boom(*args, **kwargs):
        raise NotImplementedError("nope")

    monkeypatch.setattr(me_shell, "make_radint_pointer_torch", boom)
    exp = _tiny_full_me_exp()

    with pytest.raises(NotImplementedError):
        prepare_me_shell(exp, radint_mode="torch")


def test_unknown_mode_rejected(isolated_cache):
    exp = _tiny_full_me_exp()
    with pytest.raises(ValueError, match="radint_mode"):
        prepare_me_shell(exp, radint_mode="numpy")


def test_forced_chinook_ignores_torch_cache_entry(isolated_cache):
    """A cached torch build must not satisfy an explicit chinook request."""
    assert prepare_me_shell(_tiny_full_me_exp()).radint_source == "torch"

    shell = prepare_me_shell(_tiny_full_me_exp(), radint_mode="chinook")
    assert shell.radint_source == "chinook"


def test_forced_torch_ignores_chinook_cache_entry(isolated_cache):
    """A cached chinook build must not satisfy an explicit torch request."""
    first = prepare_me_shell(_tiny_full_me_exp(), radint_mode="chinook")
    assert first.radint_source == "chinook"

    shell = prepare_me_shell(_tiny_full_me_exp(), radint_mode="torch")
    assert shell.radint_source == "torch"


def test_forced_modes_still_cache_within_a_backend(isolated_cache):
    """Backend separation must not disable caching."""
    for mode in ("chinook", "torch"):
        assert prepare_me_shell(_tiny_full_me_exp(), radint_mode=mode).radint_source == mode
        assert prepare_me_shell(_tiny_full_me_exp(), radint_mode=mode).radint_source == "cache"


def test_auto_reuses_its_own_cache_entry(isolated_cache):
    """Auto stays parity-shared with itself across calls."""
    assert prepare_me_shell(_tiny_full_me_exp()).radint_source == "torch"
    assert prepare_me_shell(_tiny_full_me_exp()).radint_source == "cache"


@pytest.mark.parametrize("rad_type", [None, "slater", "hydrogenic"])
def test_auto_selects_torch_for_analytic_rad_types(isolated_cache, rad_type):
    """No chinook fallback warning for the rad_types the torch port covers."""
    exp = _tiny_full_me_exp(rad_type)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        shell = prepare_me_shell(exp)
    assert shell.radint_source == "torch"


@pytest.mark.parametrize("rad_type", [None, "slater", "hydrogenic"])
def test_torch_and_chinook_shells_agree(isolated_cache, rad_type):
    """Both backends give the same Bfuncs on the real experiment basis."""
    exp_c = _tiny_full_me_exp(rad_type)
    exp_t = _tiny_full_me_exp(rad_type)

    shell_c = prepare_me_shell(exp_c, radint_mode="chinook", ylm_mode="chinook")
    shell_t = prepare_me_shell(exp_t, radint_mode="torch", ylm_mode="torch")

    assert shell_c.radint_source == "chinook"
    assert shell_t.radint_source == "torch"
    assert shell_c.ylm_source == "chinook"
    assert shell_t.ylm_source == "torch"

    assert np.array_equal(shell_c.radint_pointers, shell_t.radint_pointers)
    energies = np.linspace(-3.0, 3.0, 13)
    for i in range(len(shell_c.Bfuncs)):
        for j in (0, 1):
            yc = np.array([complex(shell_c.Bfuncs[i, j](e)) for e in energies])
            yt = np.array([complex(shell_t.Bfuncs[i, j](e)) for e in energies])
            np.testing.assert_allclose(yt, yc, rtol=1e-5, atol=1e-6)


def test_all_Y_torch_matches_chinook(isolated_cache):
    """Task 7 gate: torch all_Y / Gaunt tables match chinook within 1e-10."""
    from chinook.ARPES_lib import all_Y
    from grizzly.me_shell import all_Y_torch

    exp = _tiny_full_me_exp()
    exp.basis = exp.rot_basis()

    Lc, Mc, Gc, Pc = all_Y(exp.basis)
    Lt, Mt, Gt, Pt = all_Y_torch(exp.basis, device="cpu")

    np.testing.assert_array_equal(Lt, Lc)
    np.testing.assert_array_equal(Mt, Mc)
    np.testing.assert_array_equal(Pt, Pc)
    max_err = float(np.max(np.abs(Gt - Gc)))
    assert max_err < 1e-10, f"all_Y Gaunt max_err={max_err}"


def test_ylm_auto_defaults_to_torch(isolated_cache):
    shell = prepare_me_shell(_tiny_full_me_exp())
    assert shell.ylm_source == "torch"


def test_ylm_mode_chinook(isolated_cache):
    shell = prepare_me_shell(_tiny_full_me_exp(), ylm_mode="chinook")
    assert shell.ylm_source == "chinook"


def test_ylm_env_var(isolated_cache, monkeypatch):
    monkeypatch.setenv("GRIZZLY_YLM", "chinook")
    assert prepare_me_shell(_tiny_full_me_exp()).ylm_source == "chinook"


def test_ylm_auto_falls_back(isolated_cache, monkeypatch):
    import grizzly.me_shell as me_shell

    def boom(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(me_shell, "all_Y_torch", boom)
    with pytest.warns(RuntimeWarning, match="falling back to chinook"):
        shell = prepare_me_shell(_tiny_full_me_exp())
    assert shell.ylm_source == "chinook"
