"""Features deferred to GrizzlyME v2."""

from __future__ import annotations

SPIN_DEFERRED_MSG = (
    "Spin-resolved ARPES is not supported in GrizzlyME v1. "
    "Use spin={'bool': False} or plain chinook for spin calculations. "
    "Spin / SARPES support is planned for GrizzlyME v2."
)

SARPES_DEFERRED_MSG = (
    "SARPES and angle-resolved polarization rotation are not supported in "
    "GrizzlyME v1. Planned for GrizzlyME v2."
)


class GrizzlyMEv2FeatureError(NotImplementedError):
    """Raised when calling a capability reserved for GrizzlyME v2."""


def _spin_flag(obj) -> bool:
    if hasattr(obj, "spin"):
        val = obj.spin
        if isinstance(val, dict):
            return bool(val.get("bool", False))
        return bool(val)
    tb = getattr(obj, "TB", None)
    if tb is not None and hasattr(tb, "spin"):
        val = tb.spin
        if isinstance(val, dict):
            return bool(val.get("bool", False))
        return bool(val)
    return False


def require_spinless(obj, *, feature: str = "this operation") -> None:
    """Fail fast on spinful models until GrizzlyME v2."""
    if _spin_flag(obj):
        raise GrizzlyMEv2FeatureError(f"{feature}: {SPIN_DEFERRED_MSG}")
