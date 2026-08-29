"""In-memory + optional disk cache for chinook Slater radint setup."""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Hashable, Optional


def dig_range_from_cube(cube) -> tuple[float, float]:
    """Match chinook ``experiment.datacube`` energy padding for radint."""
    e0, e1, ne = cube[2][0], cube[2][1], cube[2][2]
    dE = (e1 - e0) / ne
    return (e0 - 5.0 * dE, e1 + 5.0 * dE)


def basis_fingerprint(basis) -> tuple:
    """Stable fingerprint of orbital list (pre- or post-rotation)."""
    rows = []
    for o in basis:
        rows.append(
            (
                int(o.atom),
                int(getattr(o, "Z", -1)),
                int(o.n),
                int(o.l),
                str(o.label),
                float(o.depth),
                float(o.sigma),
                bool(getattr(o, "spin", False)),
            )
        )
    return tuple(rows)


def _freeze(obj: Any) -> Hashable:
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    return repr(obj)


def make_radint_cache_key(exp) -> Optional[tuple]:
    """Key for radint reuse. Returns None if caching is unsafe (e.g. slab truncate)."""
    if getattr(exp, "truncate", False):
        return None
    cube = exp.cube
    return (
        basis_fingerprint(exp.TB.basis),
        float(exp.hv),
        float(exp.W),
        str(getattr(exp, "rad_type", "slater")).lower(),
        _freeze(getattr(exp, "rad_args", None)),
        _freeze(getattr(exp, "phase_shifts", None)),
        float(getattr(exp, "mfp", 10.0)),
        float(getattr(exp, "ang", 0.0)),
        dig_range_from_cube(cube),
    )


def default_disk_dir() -> Path:
    """``$GRIZZLY_RADINT_CACHE_DIR`` or ``~/.cache/grizzlyme/radint``."""
    env = os.environ.get("GRIZZLY_RADINT_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "grizzlyme" / "radint"


def disk_cache_enabled() -> bool:
    """Disable with ``GRIZZLY_RADINT_DISK=0`` / ``false`` / ``off``."""
    v = os.environ.get("GRIZZLY_RADINT_DISK", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _key_digest(key: tuple) -> str:
    blob = pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(blob).hexdigest()


class RadintSetupCache:
    """Memory store with optional pickle disk backend (survives process restart)."""

    def __init__(self, disk_dir: Optional[Path] = None, use_disk: Optional[bool] = None):
        self._store: dict[tuple, tuple] = {}
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.disk_writes = 0
        self.use_disk = disk_cache_enabled() if use_disk is None else bool(use_disk)
        self.disk_dir = Path(disk_dir) if disk_dir is not None else default_disk_dir()

    def _disk_path(self, key: tuple) -> Path:
        return self.disk_dir / f"{_key_digest(key)}.pkl"

    def _load_disk(self, key: tuple) -> Optional[tuple]:
        if not self.use_disk:
            return None
        path = self._disk_path(key)
        if not path.is_file():
            return None
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict) or payload.get("key") != key:
                return None
            Bfuncs = payload["Bfuncs"]
            pointers = payload["radint_pointers"]
            self._store[key] = (Bfuncs, pointers)
            self.disk_hits += 1
            return Bfuncs, pointers
        except Exception:
            return None

    def _save_disk(self, key: tuple, Bfuncs, radint_pointers) -> None:
        if not self.use_disk:
            return
        try:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            path = self._disk_path(key)
            tmp = path.with_suffix(".pkl.tmp")
            payload = {
                "key": key,
                "Bfuncs": Bfuncs,
                "radint_pointers": radint_pointers,
            }
            with tmp.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            self.disk_writes += 1
        except Exception:
            # Disk is best-effort; memory cache still works.
            try:
                tmp = self._disk_path(key).with_suffix(".pkl.tmp")
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def get(self, key: Optional[tuple]) -> Optional[tuple]:
        if key is None:
            return None
        hit = self._store.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        disk = self._load_disk(key)
        if disk is not None:
            self.hits += 1
            return disk
        self.misses += 1
        return None

    def put(self, key: Optional[tuple], Bfuncs, radint_pointers) -> None:
        if key is None:
            return
        self._store[key] = (Bfuncs, radint_pointers)
        self._save_disk(key, Bfuncs, radint_pointers)

    def clear(self, disk: bool = False) -> None:
        """Clear memory; if ``disk`` also delete pickle files in ``disk_dir``."""
        self._store.clear()
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.disk_writes = 0
        if disk and self.disk_dir.is_dir():
            for p in self.disk_dir.glob("*.pkl"):
                try:
                    p.unlink()
                except OSError:
                    pass


# Shared by all GrizzlyExperiment instances in-process (repeat cubes / benches).
DEFAULT_RADINT_CACHE = RadintSetupCache()
