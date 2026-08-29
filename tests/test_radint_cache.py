"""Radint setup cache: repeat datacube skips make_radint_pointer."""

from __future__ import annotations

import collections
import collections.abc
import copy
import time

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import numpy as np
import pytest

import chinook.radint_lib as radint_lib

from grizzly import GrizzlyExperiment
from grizzly.radint_cache import (
    DEFAULT_RADINT_CACHE,
    RadintSetupCache,
    dig_range_from_cube,
    make_radint_cache_key,
)
from tests.test_full_pipeline import build_graphene_tb_arpes


@pytest.fixture(autouse=True)
def _fresh_radint_cache(tmp_path, monkeypatch):
    """Isolate cache between tests (module default is process-wide)."""
    monkeypatch.setenv("GRIZZLY_RADINT_CACHE_DIR", str(tmp_path / "radint_mem"))
    monkeypatch.setenv("GRIZZLY_RADINT_DISK", "0")  # memory-only unit tests
    DEFAULT_RADINT_CACHE.clear(disk=False)
    DEFAULT_RADINT_CACHE.use_disk = False
    DEFAULT_RADINT_CACHE.disk_dir = tmp_path / "radint_mem"
    yield
    DEFAULT_RADINT_CACHE.clear(disk=False)


def _small_arpes(n: int = 6, n_e: int = 8):
    tb, arpes = build_graphene_tb_arpes()
    arpes = copy.deepcopy(arpes)
    arpes["cube"] = {
        "Tx": (-15.0, 15.0, n),
        "Ty": (-15.0, 15.0, n),
        "E": (-3.0, 3.0, n_e),
    }
    return tb, arpes


def test_dig_range_matches_chinook_formula():
    cube = ((-1.0, 1.0, 10), (-1.0, 1.0, 10), (-2.0, 2.0, 20))
    dE = (2.0 - (-2.0)) / 20
    assert dig_range_from_cube(cube) == (-2.0 - 5 * dE, 2.0 + 5 * dE)


def test_radint_cache_hit_skips_make_radint_pointer(monkeypatch):
    tb, arpes = _small_arpes(6, 8)
    g = GrizzlyExperiment(tb, arpes, device="cpu")

    calls = {"n": 0}
    real = radint_lib.make_radint_pointer

    def counting_make(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(radint_lib, "make_radint_pointer", counting_make)

    assert g.datacube(use_grizzly_diagonalize=False) is True
    assert calls["n"] == 1
    assert g._radint_cache.misses == 1

    arpes2 = copy.deepcopy(arpes)
    arpes2["cube"] = {
        "Tx": (-15.0, 15.0, 8),
        "Ty": (-15.0, 15.0, 8),
        "E": arpes["cube"]["E"],
    }
    g2 = GrizzlyExperiment(tb, arpes2, device="cpu")
    g2._radint_cache = g._radint_cache

    assert g2.datacube(use_grizzly_diagonalize=False) is True
    assert calls["n"] == 1, "second datacube must not call make_radint_pointer"
    assert g2._radint_cache.hits >= 1


def test_radint_cache_hit_mk_parity():
    tb, arpes = _small_arpes(6, 8)
    g1 = GrizzlyExperiment(tb, arpes, device="cpu")
    g1.datacube(use_grizzly_diagonalize=False)
    mk1 = np.copy(g1.Mk)

    g2 = GrizzlyExperiment(tb, arpes, device="cpu")
    g2._radint_cache = g1._radint_cache
    g2.datacube(use_grizzly_diagonalize=False)
    assert g2._radint_cache.hits >= 1
    max_diff = float(np.max(np.abs(g2.Mk - mk1)))
    assert max_diff < 1e-10, f"Mk after cache hit |Δ|={max_diff:.2e}"


def test_hv_change_is_cache_miss():
    tb, arpes = _small_arpes(5, 6)
    g1 = GrizzlyExperiment(tb, arpes, device="cpu")
    g1.datacube(use_grizzly_diagonalize=False)

    arpes_hv = copy.deepcopy(arpes)
    arpes_hv["hv"] = arpes["hv"] + 10.0
    g2 = GrizzlyExperiment(tb, arpes_hv, device="cpu")
    g2._radint_cache = g1._radint_cache
    assert make_radint_cache_key(g1._exp) != make_radint_cache_key(g2._exp)
    g2.datacube(use_grizzly_diagonalize=False)
    assert g2._radint_cache.misses >= 2


def test_second_datacube_faster_than_first():
    """Smoke: wall time of 2nd call (shared cache) well below 1st radint cost."""
    tb, arpes = _small_arpes(10, 10)
    g1 = GrizzlyExperiment(tb, arpes, device="cpu")
    t0 = time.perf_counter()
    g1.datacube(use_grizzly_diagonalize=False)
    t_first = time.perf_counter() - t0

    g2 = GrizzlyExperiment(tb, arpes, device="cpu")
    g2._radint_cache = g1._radint_cache
    t0 = time.perf_counter()
    g2.datacube(use_grizzly_diagonalize=False)
    t_second = time.perf_counter() - t0

    assert g2._radint_cache.hits >= 1
    assert t_second < t_first * 0.7 or t_second < 0.15
