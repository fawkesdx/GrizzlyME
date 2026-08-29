"""Disk-backed radint cache (survives process restart)."""

from __future__ import annotations

import collections
import collections.abc
import copy

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import numpy as np
import pytest

import chinook.radint_lib as radint_lib

from grizzly import GrizzlyExperiment
from grizzly.radint_cache import DEFAULT_RADINT_CACHE, RadintSetupCache
from tests.test_radint_cache import _small_arpes


@pytest.fixture(autouse=True)
def _isolate_default_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIZZLY_RADINT_CACHE_DIR", str(tmp_path / "radint"))
    monkeypatch.setenv("GRIZZLY_RADINT_DISK", "1")
    DEFAULT_RADINT_CACHE.clear(disk=False)
    DEFAULT_RADINT_CACHE.disk_dir = tmp_path / "radint"
    DEFAULT_RADINT_CACHE.use_disk = True
    DEFAULT_RADINT_CACHE.disk_hits = 0
    DEFAULT_RADINT_CACHE.disk_writes = 0
    yield
    DEFAULT_RADINT_CACHE.clear(disk=False)


def test_disk_write_and_cold_memory_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIZZLY_RADINT_CACHE_DIR", str(tmp_path / "radint"))
    cache = RadintSetupCache(disk_dir=tmp_path / "radint", use_disk=True)

    tb, arpes = _small_arpes(5, 6)
    g = GrizzlyExperiment(tb, arpes, device="cpu")
    g._radint_cache = cache
    g.datacube(use_grizzly_diagonalize=False)
    assert cache.disk_writes == 1
    assert cache.misses == 1
    mk1 = np.copy(g.Mk)

    # Simulate new process: empty memory, same disk dir
    cold = RadintSetupCache(disk_dir=tmp_path / "radint", use_disk=True)
    assert len(cold._store) == 0

    calls = {"n": 0}
    real = radint_lib.make_radint_pointer

    def counting_make(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(radint_lib, "make_radint_pointer", counting_make)

    g2 = GrizzlyExperiment(tb, arpes, device="cpu")
    g2._radint_cache = cold
    g2.datacube(use_grizzly_diagonalize=False)

    assert calls["n"] == 0
    assert cold.disk_hits == 1
    assert cold.hits == 1
    assert float(np.max(np.abs(g2.Mk - mk1))) < 1e-10


def test_disk_disabled_no_files(tmp_path):
    cache = RadintSetupCache(disk_dir=tmp_path / "radint_off", use_disk=False)
    tb, arpes = _small_arpes(5, 6)
    g = GrizzlyExperiment(tb, arpes, device="cpu")
    g._radint_cache = cache
    g.datacube(use_grizzly_diagonalize=False)
    assert cache.disk_writes == 0
    assert not (tmp_path / "radint_off").exists() or list(
        (tmp_path / "radint_off").glob("*.pkl")
    ) == []
