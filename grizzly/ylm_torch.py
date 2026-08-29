"""
PyTorch implementation of Spherical Harmonics Y_l^m(theta, phi) and Gaunt coefficients.
Vectorized over batches of angles (B) and orbital quantum numbers (N) on CPU/GPU.
Matches exact analytical definitions in chinook.Ylm for l = 0, 1, 2, 3, 4.
"""

import math
import torch
from typing import Union, Sequence


def _ensure_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        if device is not None or dtype is not None:
            return x.to(
                device=device if device is not None else x.device,
                dtype=dtype if dtype is not None else x.dtype
            )
        return x
    return torch.tensor(x, device=device, dtype=dtype)


def compute_ylm_table(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Computes all spherical harmonics Y_l^m(theta, phi) for l in [0, 4] and m in [-4, 4].
    
    Parameters
    ----------
    theta : torch.Tensor of shape (B,)
        Polar angle in radians [0, pi].
    phi : torch.Tensor of shape (B,)
        Azimuthal angle in radians [0, 2*pi].
        
    Returns
    -------
    Y_table : torch.Tensor of shape (B, 5, 9), dtype=complex
        Spherical harmonics table where Y_table[:, l, m + 4] = Y_l^m(theta, phi).
    """
    B = theta.shape[0]
    device = theta.device
    real_dtype = theta.dtype if theta.is_floating_point() else torch.float64
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    
    theta = theta.to(dtype=real_dtype)
    phi = phi.to(dtype=real_dtype)
    
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    cos2 = cos_th ** 2
    cos3 = cos_th ** 3
    cos4 = cos_th ** 4
    sin2 = sin_th ** 2
    sin3 = sin_th ** 3
    sin4 = sin_th ** 4
    
    pi = math.pi
    
    # Initialize output table (B, 5, 9)
    # l index: 0..4 (dim 1)
    # m index: m + 4 in 0..8 (dim 2)
    Y_table = torch.zeros((B, 5, 9), dtype=complex_dtype, device=device)
    
    # exp(i * m * phi) for m in [-4, 4]
    m_range = torch.arange(-4, 5, device=device, dtype=real_dtype)  # (9,)
    phases = torch.polar(
        torch.ones((B, 9), dtype=real_dtype, device=device),
        phi.unsqueeze(1) * m_range.unsqueeze(0)
    )
    
    # l = 0
    # m = 0
    c00 = 0.5 * math.sqrt(1.0 / pi)
    Y_table[:, 0, 0 + 4] = c00 * phases[:, 0 + 4]
    
    # l = 1
    # m = 0
    c10 = 0.5 * math.sqrt(3.0 / pi)
    Y_table[:, 1, 0 + 4] = (c10 * cos_th).to(complex_dtype)
    
    # |m| = 1
    c11 = 0.5 * math.sqrt(3.0 / (2.0 * pi))
    Y_table[:, 1, 1 + 4] = (-c11 * sin_th).to(complex_dtype) * phases[:, 1 + 4]
    Y_table[:, 1, -1 + 4] = (c11 * sin_th).to(complex_dtype) * phases[:, -1 + 4]
    
    # l = 2
    # m = 0
    c20 = 0.25 * math.sqrt(5.0 / pi)
    Y_table[:, 2, 0 + 4] = (c20 * (3.0 * cos2 - 1.0)).to(complex_dtype)
    
    # |m| = 1
    c21 = 0.5 * math.sqrt(15.0 / (2.0 * pi))
    Y_table[:, 2, 1 + 4] = (-c21 * sin_th * cos_th).to(complex_dtype) * phases[:, 1 + 4]
    Y_table[:, 2, -1 + 4] = (c21 * sin_th * cos_th).to(complex_dtype) * phases[:, -1 + 4]
    
    # |m| = 2
    c22 = 0.25 * math.sqrt(15.0 / (2.0 * pi))
    Y_table[:, 2, 2 + 4] = (c22 * sin2).to(complex_dtype) * phases[:, 2 + 4]
    Y_table[:, 2, -2 + 4] = (c22 * sin2).to(complex_dtype) * phases[:, -2 + 4]
    
    # l = 3
    # m = 0
    c30 = 0.25 * math.sqrt(7.0 / pi)
    Y_table[:, 3, 0 + 4] = (c30 * (5.0 * cos3 - 3.0 * cos_th)).to(complex_dtype)
    
    # |m| = 1
    c31 = (1.0 / 8.0) * math.sqrt(21.0 / pi)
    Y_table[:, 3, 1 + 4] = (-c31 * sin_th * (5.0 * cos2 - 1.0)).to(complex_dtype) * phases[:, 1 + 4]
    Y_table[:, 3, -1 + 4] = (c31 * sin_th * (5.0 * cos2 - 1.0)).to(complex_dtype) * phases[:, -1 + 4]
    
    # |m| = 2
    c32 = 0.25 * math.sqrt(105.0 / (2.0 * pi))
    Y_table[:, 3, 2 + 4] = (c32 * sin2 * cos_th).to(complex_dtype) * phases[:, 2 + 4]
    Y_table[:, 3, -2 + 4] = (c32 * sin2 * cos_th).to(complex_dtype) * phases[:, -2 + 4]
    
    # |m| = 3
    c33 = (1.0 / 8.0) * math.sqrt(35.0 / pi)
    Y_table[:, 3, 3 + 4] = (-c33 * sin3).to(complex_dtype) * phases[:, 3 + 4]
    Y_table[:, 3, -3 + 4] = (c33 * sin3).to(complex_dtype) * phases[:, -3 + 4]
    
    # l = 4
    # m = 0
    c40 = (3.0 / 16.0) * math.sqrt(1.0 / pi)
    Y_table[:, 4, 0 + 4] = (c40 * (35.0 * cos4 - 30.0 * cos2 + 3.0)).to(complex_dtype)
    
    # |m| = 1
    c41 = (3.0 / 8.0) * math.sqrt(5.0 / pi)
    Y_table[:, 4, 1 + 4] = (-c41 * sin_th * (7.0 * cos3 - 3.0 * cos_th)).to(complex_dtype) * phases[:, 1 + 4]
    Y_table[:, 4, -1 + 4] = (c41 * sin_th * (7.0 * cos3 - 3.0 * cos_th)).to(complex_dtype) * phases[:, -1 + 4]
    
    # |m| = 2
    c42 = (3.0 / 8.0) * math.sqrt(5.0 / (2.0 * pi))
    Y_table[:, 4, 2 + 4] = (c42 * sin2 * (7.0 * cos2 - 1.0)).to(complex_dtype) * phases[:, 2 + 4]
    Y_table[:, 4, -2 + 4] = (c42 * sin2 * (7.0 * cos2 - 1.0)).to(complex_dtype) * phases[:, -2 + 4]
    
    # |m| = 3
    c43 = (3.0 / 8.0) * math.sqrt(35.0 / pi)
    Y_table[:, 4, 3 + 4] = (-c43 * sin3 * cos_th).to(complex_dtype) * phases[:, 3 + 4]
    Y_table[:, 4, -3 + 4] = (c43 * sin3 * cos_th).to(complex_dtype) * phases[:, -3 + 4]
    
    # |m| = 4
    c44 = (3.0 / 16.0) * math.sqrt(35.0 / (2.0 * pi))
    Y_table[:, 4, 4 + 4] = (c44 * sin4).to(complex_dtype) * phases[:, 4 + 4]
    Y_table[:, 4, -4 + 4] = (c44 * sin4).to(complex_dtype) * phases[:, -4 + 4]
    
    return Y_table


def batched_Y(
    l_arr: Union[torch.Tensor, Sequence[int]],
    m_arr: Union[torch.Tensor, Sequence[int]],
    theta: torch.Tensor,
    phi: torch.Tensor,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """
    Batched calculation of spherical harmonics Y_l^m(theta, phi).
    
    Parameters
    ----------
    l_arr : 1D sequence or tensor of length N
        Orbital angular momentum quantum numbers (0 <= l <= 4).
    m_arr : 1D sequence or tensor of length N
        Azimuthal angular momentum quantum numbers (|m| <= l).
    theta : 1D tensor of shape (B,)
        Polar angles in radians [0, pi].
    phi : 1D tensor of shape (B,)
        Azimuthal angles in radians [0, 2*pi].
    device : str or torch.device
        Device on which computation is performed.
    dtype : torch.dtype
        Floating point dtype (default torch.float64).
        
    Returns
    -------
    Y : torch.Tensor of shape (B, N), dtype=complex (complex128 or complex64)
        Spherical harmonic values for each angle and (l, m) pair.
    """
    target_device = torch.device(device) if isinstance(device, str) else device
    
    theta = _ensure_tensor(theta, device=target_device, dtype=dtype)
    phi = _ensure_tensor(phi, device=target_device, dtype=dtype)
    
    if theta.ndim == 0:
        theta = theta.unsqueeze(0)
    if phi.ndim == 0:
        phi = phi.unsqueeze(0)
        
    l_tensor = _ensure_tensor(l_arr, device=target_device, dtype=torch.long)
    m_tensor = _ensure_tensor(m_arr, device=target_device, dtype=torch.long)
    
    if l_tensor.ndim == 0:
        l_tensor = l_tensor.unsqueeze(0)
    if m_tensor.ndim == 0:
        m_tensor = m_tensor.unsqueeze(0)
        
    # Validity mask: 0 <= l <= 4 and |m| <= l
    valid_mask = (l_tensor >= 0) & (l_tensor <= 4) & (torch.abs(m_tensor) <= l_tensor)
    
    # Compute full Ylm table: shape (B, 5, 9)
    Y_table = compute_ylm_table(theta, phi)
    
    # Safe indices for lookup
    safe_l = torch.clamp(l_tensor, 0, 4)
    safe_m = torch.clamp(m_tensor, -4, 4) + 4
    
    # Advanced indexing: (B, 5, 9) -> (B, N)
    Y_out = Y_table[:, safe_l, safe_m]
    
    # Zero out invalid (l, m) entries
    if not valid_mask.all():
        Y_out = Y_out * valid_mask.unsqueeze(0).to(Y_out.dtype)
        
    return Y_out


def batched_gaunt(
    l_arr: Union[torch.Tensor, Sequence[int]],
    m_arr: Union[torch.Tensor, Sequence[int]],
    dl_arr: Union[torch.Tensor, Sequence[int]],
    dm_arr: Union[torch.Tensor, Sequence[int]],
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """
    Batched calculation of Gaunt coefficients for dipole-allowed transitions.
    Vectorizes chinook.Ylm.gaunt.
    
    Parameters
    ----------
    l_arr : sequence or tensor of length N
        Initial orbital angular momentum quantum numbers.
    m_arr : sequence or tensor of length N
        Initial azimuthal angular momentum quantum numbers.
    dl_arr : sequence or tensor of length N
        Change in l (+1 or -1).
    dm_arr : sequence or tensor of length N
        Change in m (+1, 0, or -1).
    device : str or torch.device
        Device for tensor placement.
    dtype : torch.dtype
        Float dtype for the returned coefficients.
        
    Returns
    -------
    gaunt_coeff : torch.Tensor of shape (N,), dtype=dtype
        Gaunt coefficients for dipole transition matrix elements.
    """
    target_device = torch.device(device) if isinstance(device, str) else device
    
    l = _ensure_tensor(l_arr, device=target_device, dtype=torch.long)
    m = _ensure_tensor(m_arr, device=target_device, dtype=torch.long)
    dl = _ensure_tensor(dl_arr, device=target_device, dtype=torch.long)
    dm = _ensure_tensor(dm_arr, device=target_device, dtype=torch.long)
    
    out = torch.zeros(l.shape, dtype=dtype, device=target_device)
    
    # Parity factors (-1)^k
    phase_m = torch.where((m.abs() % 2) == 0, 1.0, -1.0)
    phase_m1 = -phase_m
    
    lf = l.to(dtype)
    mf = m.to(dtype)
    pi = math.pi
    
    # Selection rule mask
    valid = (torch.abs(m + dm) <= (l + dl)) & (l + dl >= 0) & (l >= 0) & (torch.abs(m) <= l)
    
    # dl = +1 cases
    mask_dl_plus1 = valid & (dl == 1)
    
    m11 = mask_dl_plus1 & (dm == 1)
    if m11.any():
        num = 3.0 * (lf[m11] + mf[m11] + 2.0) * (lf[m11] + mf[m11] + 1.0)
        den = 8.0 * pi * (2.0 * lf[m11] + 3.0) * (2.0 * lf[m11] + 1.0)
        out[m11] = phase_m1[m11].to(dtype) * torch.sqrt(num / den)
        
    m10 = mask_dl_plus1 & (dm == 0)
    if m10.any():
        num = 3.0 * (lf[m10] - mf[m10] + 1.0) * (lf[m10] + mf[m10] + 1.0)
        den = 4.0 * pi * (2.0 * lf[m10] + 3.0) * (2.0 * lf[m10] + 1.0)
        out[m10] = phase_m[m10].to(dtype) * torch.sqrt(num / den)
        
    m1_minus1 = mask_dl_plus1 & (dm == -1)
    if m1_minus1.any():
        num = 3.0 * (lf[m1_minus1] - mf[m1_minus1] + 2.0) * (lf[m1_minus1] - mf[m1_minus1] + 1.0)
        den = 8.0 * pi * (2.0 * lf[m1_minus1] + 3.0) * (2.0 * lf[m1_minus1] + 1.0)
        out[m1_minus1] = phase_m1[m1_minus1].to(dtype) * torch.sqrt(num / den)
        
    # dl = -1 cases
    mask_dl_minus1 = valid & (dl == -1)
    
    mm11 = mask_dl_minus1 & (dm == 1)
    if mm11.any():
        num = 3.0 * (lf[mm11] - mf[mm11]) * (lf[mm11] - mf[mm11] - 1.0)
        den = 8.0 * pi * (2.0 * lf[mm11] + 1.0) * (2.0 * lf[mm11] - 1.0)
        out[mm11] = phase_m[mm11].to(dtype) * torch.sqrt(num / den)
        
    mm10 = mask_dl_minus1 & (dm == 0)
    if mm10.any():
        num = 3.0 * (lf[mm10] + mf[mm10]) * (lf[mm10] - mf[mm10])
        den = 4.0 * pi * (2.0 * lf[mm10] + 1.0) * (2.0 * lf[mm10] - 1.0)
        out[mm10] = phase_m[mm10].to(dtype) * torch.sqrt(num / den)
        
    mm1_minus1 = mask_dl_minus1 & (dm == -1)
    if mm1_minus1.any():
        num = 3.0 * (lf[mm1_minus1] + mf[mm1_minus1]) * (lf[mm1_minus1] + mf[mm1_minus1] - 1.0)
        den = 8.0 * pi * (2.0 * lf[mm1_minus1] + 1.0) * (2.0 * lf[mm1_minus1] - 1.0)
        out[mm1_minus1] = phase_m[mm1_minus1].to(dtype) * torch.sqrt(num / den)
        
    return out
