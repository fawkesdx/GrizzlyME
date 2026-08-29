"""Headless chinook import (no _tkinter required)."""

from grizzly import GrizzlyExperiment


def test_grizzly_imports_without_requiring_tk():
    assert GrizzlyExperiment is not None
