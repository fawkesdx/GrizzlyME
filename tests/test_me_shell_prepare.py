"""Tests for reusable matrix-element shell preparation."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from grizzly import GrizzlyExperiment, prepare_me_shell
from grizzly.radint_cache import RadintSetupCache


def _tiny_full_me_exp():
    helpers = runpy.run_path(
        str(Path(__file__).with_name("test_full_pipeline.py"))
    )
    tb, arpes_dict = helpers["build_graphene_tb_arpes"]()
    return GrizzlyExperiment(tb, arpes_dict, device="cpu")


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    import grizzly.me_shell as me_shell

    cache = RadintSetupCache(disk_dir=tmp_path, use_disk=False)
    monkeypatch.setattr(me_shell, "DEFAULT_RADINT_CACHE", cache)
    monkeypatch.delenv("GRIZZLY_RADINT", raising=False)
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


def test_torch_and_chinook_shells_agree(isolated_cache):
    """Both backends give the same Bfuncs on the real experiment basis."""
    exp_c = _tiny_full_me_exp()
    exp_t = _tiny_full_me_exp()

    shell_c = prepare_me_shell(exp_c, radint_mode="chinook")
    isolated_cache.clear()
    shell_t = prepare_me_shell(exp_t, radint_mode="torch")

    assert np.array_equal(shell_c.radint_pointers, shell_t.radint_pointers)
    energies = np.linspace(-3.0, 3.0, 13)
    for i in range(len(shell_c.Bfuncs)):
        for j in (0, 1):
            yc = np.array([complex(shell_c.Bfuncs[i, j](e)) for e in energies])
            yt = np.array([complex(shell_t.Bfuncs[i, j](e)) for e in energies])
            np.testing.assert_allclose(yt, yc, rtol=1e-5, atol=1e-6)
