"""Tests for reusable matrix-element shell preparation."""

from __future__ import annotations

import runpy
from pathlib import Path

from grizzly import GrizzlyExperiment, prepare_me_shell
from grizzly.radint_cache import RadintSetupCache


def _tiny_full_me_exp():
    helpers = runpy.run_path(
        str(Path(__file__).with_name("test_full_pipeline.py"))
    )
    tb, arpes_dict = helpers["build_graphene_tb_arpes"]()
    return GrizzlyExperiment(tb, arpes_dict, device="cpu")


def test_prepare_me_shell_second_call_is_cache(monkeypatch, tmp_path):
    import grizzly.me_shell as me_shell

    cache = RadintSetupCache(disk_dir=tmp_path, use_disk=False)
    monkeypatch.setattr(me_shell, "DEFAULT_RADINT_CACHE", cache)
    exp = _tiny_full_me_exp()

    s1 = prepare_me_shell(exp)
    s2 = prepare_me_shell(exp)

    assert s1.radint_source == "chinook"
    assert s2.radint_source == "cache"
    assert len(s1.radint_pointers) == len(s1.basis)


def test_prepare_me_shell_torch_mode_is_phase_a_placeholder():
    exp = _tiny_full_me_exp()

    try:
        prepare_me_shell(exp, radint_mode="torch")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("radint_mode='torch' must remain unavailable")
