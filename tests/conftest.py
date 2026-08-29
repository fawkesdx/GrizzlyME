"""Pytest hooks for GrizzlyME (chinook compat on Python 3.10+)."""

import collections
import collections.abc

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable
