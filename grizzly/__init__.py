"""GrizzlyME — GPU-accelerated ARPES matrix elements and spectral assembly."""

import collections
import collections.abc

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

from .future import GrizzlyMEv2FeatureError, require_spinless
from .experiment import GrizzlyExperiment
from .engine import compute_all_Mk
from .hamiltonian import solve_H, extract_hopping_data, build_and_diagonalize
from .spectral import (
    compute_fermi_distribution,
    compute_M_factor,
    compute_spectral_intensity,
    gaussian_convolution_3d,
)
from .utils import get_device

__all__ = [
    "GrizzlyExperiment",
    "GrizzlyMEv2FeatureError",
    "require_spinless",
    "compute_all_Mk",
    "solve_H",
    "extract_hopping_data",
    "build_and_diagonalize",
    "compute_fermi_distribution",
    "compute_M_factor",
    "compute_spectral_intensity",
    "gaussian_convolution_3d",
    "get_device",
]
