"""
Unit tests comparing grizzly.ylm_torch functions against chinook.Ylm.
"""

import math
import numpy as np
import pytest
import torch

import chinook.Ylm as cYlm
from grizzly.ylm_torch import batched_Y, batched_gaunt


def test_batched_Y_all_lm_random_angles():
    """
    Test batched_Y against chinook.Ylm.Y for 100 random angles and all
    25 combinations of l in [0, 4] and |m| <= l.
    """
    np.random.seed(42)
    B = 100
    theta_np = np.random.uniform(0.0, np.pi, size=B)
    phi_np = np.random.uniform(0.0, 2.0 * np.pi, size=B)
    
    theta_torch = torch.tensor(theta_np, dtype=torch.float64)
    phi_torch = torch.tensor(phi_np, dtype=torch.float64)
    
    # Collect all valid (l, m) pairs
    lm_pairs = [(l, m) for l in range(5) for m in range(-l, l + 1)]
    l_arr = [p[0] for p in lm_pairs]
    m_arr = [p[1] for p in lm_pairs]
    
    # Compute using PyTorch batched_Y
    Y_torch = batched_Y(l_arr, m_arr, theta_torch, phi_torch, device='cpu', dtype=torch.float64)
    Y_torch_np = Y_torch.detach().cpu().numpy()
    
    # Compute reference using chinook.Ylm.Y
    Y_ref = np.zeros((B, len(lm_pairs)), dtype=complex)
    for b in range(B):
        for idx, (l, m) in enumerate(lm_pairs):
            Y_ref[b, idx] = cYlm.Y(l, m, theta_np[b], phi_np[b])
            
    max_err = np.max(np.abs(Y_torch_np - Y_ref))
    assert max_err < 1e-12, f"batched_Y max error ({max_err}) exceeded tolerance 1e-12"


def test_batched_Y_edge_cases():
    """
    Test boundary angles: theta = 0, pi/2, pi; phi = 0, pi, 2*pi.
    """
    thetas = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi]
    phis = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi]
    
    grid_th, grid_ph = np.meshgrid(thetas, phis)
    theta_flat = grid_th.flatten()
    phi_flat = grid_ph.flatten()
    
    theta_t = torch.tensor(theta_flat, dtype=torch.float64)
    phi_t = torch.tensor(phi_flat, dtype=torch.float64)
    
    lm_pairs = [(l, m) for l in range(5) for m in range(-l, l + 1)]
    l_arr = [p[0] for p in lm_pairs]
    m_arr = [p[1] for p in lm_pairs]
    
    Y_torch = batched_Y(l_arr, m_arr, theta_t, phi_t, dtype=torch.float64).numpy()
    
    for b in range(len(theta_flat)):
        for idx, (l, m) in enumerate(lm_pairs):
            ref_val = cYlm.Y(l, m, theta_flat[b], phi_flat[b])
            assert np.isclose(Y_torch[b, idx], ref_val, atol=1e-12), (
                f"Mismatch at theta={theta_flat[b]}, phi={phi_flat[b]}, l={l}, m={m}"
            )


def test_batched_Y_invalid_lm():
    """
    Verify that invalid l (>4 or <0) or |m| > l return zeros.
    """
    theta = torch.tensor([0.5, 1.0], dtype=torch.float64)
    phi = torch.tensor([0.2, 0.4], dtype=torch.float64)
    
    # Invalid combinations: l=5, m=2; l=2, m=3; l=1, m=-2; l=-1, m=0
    l_invalid = [5, 2, 1, -1]
    m_invalid = [2, 3, -2, 0]
    
    Y = batched_Y(l_invalid, m_invalid, theta, phi)
    assert torch.all(Y == 0.0)


def test_batched_gaunt_against_chinook():
    """
    Test batched_gaunt against chinook.Ylm.gaunt for exhaustive sweep of
    l in [0, 5], |m| <= l, dl in [-1, 1], dm in [-1, 0, 1].
    """
    l_list, m_list, dl_list, dm_list, ref_list = [], [], [], [], []
    
    for l in range(6):
        for m in range(-l, l + 1):
            for dl in (-1, 1):
                for dm in (-1, 0, 1):
                    l_list.append(l)
                    m_list.append(m)
                    dl_list.append(dl)
                    dm_list.append(dm)
                    ref_list.append(cYlm.gaunt(l, m, dl, dm))
                    
    ref_arr = np.array(ref_list, dtype=np.float64)
    
    gaunt_torch = batched_gaunt(l_list, m_list, dl_list, dm_list, device='cpu', dtype=torch.float64)
    gaunt_torch_np = gaunt_torch.detach().cpu().numpy()
    
    max_err = np.max(np.abs(gaunt_torch_np - ref_arr))
    assert max_err < 1e-12, f"batched_gaunt max error ({max_err}) exceeded tolerance 1e-12"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
