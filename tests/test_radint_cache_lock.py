from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

from grizzly.radint_cache import RadintSetupCache


def _builder_counting(counter_path: Path):
    def _build():
        # Simulate expensive radint
        time.sleep(0.3)
        n = int(counter_path.read_text() or "0")
        counter_path.write_text(str(n + 1))
        Bfuncs = ["fake"]
        pointers = [0]
        return Bfuncs, pointers

    return _build


def _worker(disk_dir: str, key, counter: str):
    cache = RadintSetupCache(disk_dir=Path(disk_dir), use_disk=True)
    cache.get_or_build(key, _builder_counting(Path(counter)))


def test_get_or_build_only_one_builder_across_two_processes(tmp_path: Path):
    counter = tmp_path / "builds.txt"
    counter.write_text("0")
    key = ("lock-test", 1.0, "hydrogenic")
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(str(tmp_path), key, str(counter)))
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    assert counter.read_text() == "1"
