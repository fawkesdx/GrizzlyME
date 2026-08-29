"""
Unit tests comparing grizzly.engine (batched GPU/CPU matrix elements)
against chinook's serial_Mk.
"""

import collections
import collections.abc
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

import numpy as np
import pytest
import torch

import chinook.build_lib as build_lib
import chinook.ARPES_lib as arpes_lib
from grizzly.engine import compute_all_Mk, batched_matrix_elements, prepare_static_tensors


def create_graphene_experiment():
    """
    Constructs a 2-orbital graphene tight-binding model (2pz on C1, C2)
    and initializes a chinook ARPES experiment with datacube precomputed.
    """
    basis_args = {
        'atoms': [0, 1],
        'Z': {0: 6, 1: 6},
        'pos': [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.42, 0.0])],
        'orbs': [['21z'], ['21z']],
        'spin': {'bool': False}
    }
    basis_dict = build_lib.gen_basis(basis_args)
    
    # Hexagonal lattice vectors (Angstroms)
    a_lattice = np.array([
        [2.46, 0.0, 0.0],
        [-1.23, 2.130446, 0.0],
        [0.0, 0.0, 20.0]
    ])
    
    # Nearest-neighbor hopping t = -2.8 eV
    hopping_list = [
        [0, 0, 0.0, 0.0, 0.0, 0.0],
        [1, 1, 0.0, 0.0, 0.0, 0.0],
        [0, 1, 0.0, 1.42, 0.0, -2.8],
        [0, 1, 1.23, -0.71, 0.0, -2.8],
        [0, 1, -1.23, -0.71, 0.0, -2.8]
    ]
    
    hamiltonian_dict = {
        'type': 'list',
        'list': hopping_list,
        'a': a_lattice.tolist(),
        'cutoff': 5.0,
        'spin': {'bool': False}
    }
    
    tb_model = build_lib.gen_TB(basis_dict, hamiltonian_dict)
    
    arpes_dict = {
        'cube': {
            'Tx': (-15.0, 15.0, 10),
            'Ty': (-15.0, 15.0, 10),
            'E': (-3.0, 3.0, 15)
        },
        'hv': 21.2,
        'W': 4.5,
        'pol': np.array([1.0, 0.0, 0.0]),
        'T': 300.0,
        'resolution': {'E': 0.05, 'k': 0.02},
        'SE': ['constant', 0.1]
    }
    
    exp = arpes_lib.experiment(tb_model, arpes_dict)
    exp.datacube()
    return exp


def create_multiorbital_experiment():
    """
    Constructs a multi-orbital model (s, px, py, pz) to test
    multiple angular momenta (l=0, l=1) and projections.
    """
    basis_args = {
        'atoms': [0, 1],
        'Z': {0: 14, 1: 14},
        'pos': [np.array([0.0, 0.0, 0.0]), np.array([1.35, 1.35, 1.35])],
        'orbs': [['30', '31x', '31y', '31z'], ['30', '31x', '31y', '31z']],
        'spin': {'bool': False}
    }
    basis_dict = build_lib.gen_basis(basis_args)
    
    a_lattice = 5.43 * np.eye(3)
    hopping_list = []
    for i in range(4):
        e_on = -5.0 if i == 0 else 0.0
        hopping_list.append([i, i, 0.0, 0.0, 0.0, e_on])
        hopping_list.append([i + 4, i + 4, 0.0, 0.0, 0.0, e_on])
    
    hopping_list.append([0, 4, 1.35, 1.35, 1.35, -2.0])
    hopping_list.append([1, 5, 1.35, 1.35, 1.35, 1.5])
    hopping_list.append([2, 6, 1.35, 1.35, 1.35, 1.5])
    hopping_list.append([3, 7, 1.35, 1.35, 1.35, 1.5])
    
    hamiltonian_dict = {
        'type': 'list',
        'list': hopping_list,
        'a': a_lattice.tolist(),
        'cutoff': 6.0,
        'spin': {'bool': False}
    }
    
    tb_model = build_lib.gen_TB(basis_dict, hamiltonian_dict)
    
    arpes_dict = {
        'cube': {
            'Tx': (-10.0, 10.0, 8),
            'Ty': (-10.0, 10.0, 8),
            'E': (-6.0, 2.0, 10)
        },
        'hv': 30.0,
        'W': 4.5,
        'pol': np.array([0.0, 1.0, 0.0]),
        'T': 100.0,
        'resolution': {'E': 0.05, 'k': 0.02},
        'SE': ['constant', 0.1]
    }
    
    exp = arpes_lib.experiment(tb_model, arpes_dict)
    exp.datacube()
    return exp


def create_spin_polarized_experiment():
    """
    Constructs a spin-polarized model to test spin-resolved matrix elements.
    """
    basis_args = {
        'atoms': [0],
        'Z': {0: 6},
        'pos': [np.array([0.0, 0.0, 0.0])],
        'orbs': [['21z']],
        'spin': {'bool': True, 'soc': False, 'lam': {0: 0.0}}
    }
    basis_dict = build_lib.gen_basis(basis_args)
    
    a_lattice = 3.0 * np.eye(3)
    hopping_list = [
        [0, 0, 0.0, 0.0, 0.0, 0.0],
        [0, 0, 3.0, 0.0, 0.0, -1.0]
    ]
    
    hamiltonian_dict = {
        'type': 'list',
        'list': hopping_list,
        'a': a_lattice.tolist(),
        'cutoff': 4.0,
        'spin': {'bool': True, 'soc': False, 'lam': {0: 0.0}}
    }
    
    tb_model = build_lib.gen_TB(basis_dict, hamiltonian_dict)
    
    arpes_dict = {
        'cube': {
            'Tx': (-10.0, 10.0, 5),
            'Ty': (-10.0, 10.0, 5),
            'E': (-2.0, 2.0, 10)
        },
        'hv': 25.0,
        'W': 4.5,
        'pol': np.array([1.0, 0.0, 0.0]),
        'T': 50.0,
        'resolution': {'E': 0.05, 'k': 0.02},
        'SE': ['constant', 0.1]
    }
    
    exp = arpes_lib.experiment(tb_model, arpes_dict)
    exp.datacube()
    return exp


def test_compute_all_Mk_graphene():
    """
    Test compute_all_Mk against chinook's serial_Mk on a 2-orbital graphene model.
    """
    exp = create_graphene_experiment()
    
    # Matrix elements already computed in datacube by serial_Mk
    Mk_chinook = np.copy(exp.Mk)
    
    # Compute with GrizzlyME engine
    Mk_grizzly = compute_all_Mk(exp, device='cpu')
    
    # Verify shape and non-zero
    assert Mk_grizzly.shape == Mk_chinook.shape
    assert np.any(np.abs(Mk_chinook) > 0.0)
    
    # Compare with strict tolerance
    max_diff = np.max(np.abs(Mk_grizzly - Mk_chinook))
    assert max_diff < 1e-6, f"Max difference ({max_diff}) exceeds tolerance 1e-6"
    assert np.allclose(Mk_grizzly, Mk_chinook, atol=1e-6)


def test_compute_all_Mk_batch_sizes():
    """
    Verify that compute_all_Mk produces identical results across different batch sizes.
    """
    exp = create_graphene_experiment()
    Mk_ref = np.copy(exp.Mk)
    
    for bs in [1, 3, 10, 100, 1000]:
        Mk_bs = compute_all_Mk(exp, batch_size=bs, device='cpu')
        assert np.allclose(Mk_bs, Mk_ref, atol=1e-6), f"Batch size {bs} produced mismatch"


def test_compute_all_Mk_multiorbital():
    """
    Test compute_all_Mk with multi-orbital basis (s and p orbitals).
    """
    exp = create_multiorbital_experiment()
    Mk_chinook = np.copy(exp.Mk)
    
    Mk_grizzly = compute_all_Mk(exp, device='cpu')
    
    assert Mk_grizzly.shape == Mk_chinook.shape
    assert np.any(np.abs(Mk_chinook) > 0.0)
    
    max_diff = np.max(np.abs(Mk_grizzly - Mk_chinook))
    assert max_diff < 1e-6, f"Multi-orbital max diff ({max_diff}) exceeds tolerance 1e-6"
    assert np.allclose(Mk_grizzly, Mk_chinook, atol=1e-6)


def test_compute_all_Mk_spin_deferred_to_v2():
    """Spin ME parity deferred — compute_all_Mk raises until GrizzlyME v2."""
    from grizzly import GrizzlyMEv2FeatureError

    exp = create_spin_polarized_experiment()
    with pytest.raises(GrizzlyMEv2FeatureError, match="GrizzlyME v2"):
        compute_all_Mk(exp, device="cpu")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
