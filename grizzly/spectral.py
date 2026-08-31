"""Spectral assembly helpers (Phase 4)."""

import numpy as np
import torch

# chinook ARPES_lib constants (T_distribution / con_ferm)
_KB = 1.38e-23
_Q = 1.602e-19
_OVERFLOW_EKBT = 709


def compute_fermi_distribution(omega: torch.Tensor, T: float) -> torch.Tensor:
    """Fermi-Dirac at mu=0; parity with chinook experiment.T_distribution."""
    omega = omega.to(dtype=torch.float64)
    ekbt = omega / (_KB * T / _Q)
    fermi = torch.zeros_like(ekbt)
    mask = ekbt < _OVERFLOW_EKBT
    fermi[mask] = 1.0 / (torch.exp(ekbt[mask]) + 1.0)
    return fermi


def compute_M_factor(Mk, pol_sph, sarpes=None) -> torch.Tensor:
    """Spinless |M·ε|² factor; chinook spectral parity.

    Parameters
    ----------
    Mk : (B, 2, 3) complex
    pol_sph : (3,) or (B, 3) complex
        Fixed spherical pol (``coord_type='momentum'``) or per-peak rotated
        pol from ``experiment.gen_all_pol()`` (``coord_type='angle'``).
    """
    if pol_sph.ndim == 1:
        contracted = torch.einsum("ijk,k->ij", Mk, pol_sph)
    else:
        contracted = torch.einsum("ijk,ik->ij", Mk, pol_sph)
    return torch.sum(torch.abs(contracted) ** 2, dim=1).to(dtype=torch.float64)


def _se_with_offset(SE, Nw: int, device, dtype=torch.complex128):
    """Apply chinook -0.00005j offset; scalar or per-energy SE."""
    offset = torch.tensor(-0.00005j, dtype=dtype, device=device)
    if isinstance(SE, (complex, float, int)):
        return torch.tensor(SE, dtype=dtype, device=device) + offset
    se = SE.to(device=device, dtype=dtype)
    if se.ndim == 0:
        return se + offset
    return se + offset


def compute_spectral_intensity(M_factor, pks, omega, SE, fermi, grid_shape) -> torch.Tensor:
    """Scatter-add peak contributions; chinook Green's function parity."""
    Ny, Nx, Nw = grid_shape
    device = omega.device
    omega = omega.to(dtype=torch.float64)
    fermi = fermi.to(dtype=torch.float64)
    M_factor = M_factor.to(dtype=torch.float64)

    ky_idx = pks[:, 1].real.to(dtype=torch.long)
    kx_idx = pks[:, 2].real.to(dtype=torch.long)
    E_p = pks[:, 3].real.to(dtype=torch.float64)

    se = _se_with_offset(SE, Nw, device)
    if se.ndim == 0:
        denom = omega.unsqueeze(0) - E_p.unsqueeze(1) - se
    else:
        denom = omega.unsqueeze(0) - E_p.unsqueeze(1) - se.unsqueeze(0)

    green_im = torch.imag(-1.0 / (torch.pi * denom))
    contrib = M_factor.unsqueeze(1) * green_im * fermi.unsqueeze(0)

    linear_idx = ky_idx * Nx + kx_idx
    I_flat = torch.zeros(Ny * Nx, Nw, dtype=torch.float64, device=device)
    index = linear_idx.unsqueeze(1).expand(-1, Nw)
    I_flat.scatter_add_(0, index, contrib)

    return I_flat.view(Ny, Nx, Nw)


def gaussian_convolution_3d(I, sigma_y, sigma_x, sigma_w) -> torch.Tensor:
    """3D Gaussian broadening via FFT; sigmas in pixel units (scipy parity)."""
    I = I.to(dtype=torch.float64)
    F = torch.fft.fftn(I)
    for dim, sigma in enumerate((sigma_y, sigma_x, sigma_w)):
        if sigma == 0:
            continue
        n = I.shape[dim]
        freq = torch.fft.fftfreq(n, d=1.0, device=I.device, dtype=torch.float64)
        gauss = torch.exp(-2.0 * (torch.pi * freq * sigma) ** 2)
        shape = [1, 1, 1]
        shape[dim] = n
        F = F * gauss.view(shape)
    return torch.fft.ifftn(F).real


def build_raw_I_from_experiment(exp, device: str | torch.device = "cpu") -> torch.Tensor:
    """Rebuild chinook raw I from experiment fields (smoke-test helper)."""
    from chinook.ARPES_lib import pol_2_sph

    from grizzly.utils import get_device

    dev = get_device(device) if isinstance(device, str) else device
    Mk = torch.as_tensor(exp.Mk, dtype=torch.complex128, device=dev)
    if getattr(exp, "coord_type", "momentum") == "angle":
        # Chinook rotates pol with cryostat angles (Tx/Ty cubes).
        pol_sph = torch.as_tensor(exp.gen_all_pol(), dtype=torch.complex128, device=dev)
    else:
        pol_sph = torch.as_tensor(pol_2_sph(exp.pol), dtype=torch.complex128, device=dev)
    M_factor = compute_M_factor(Mk, pol_sph)
    omega = torch.linspace(*exp.cube[2], dtype=torch.float64, device=dev)
    fermi = compute_fermi_distribution(omega, exp.T)
    pks = torch.as_tensor(exp.pks, dtype=torch.float64, device=dev)
    grid_shape = (exp.cube[1][2], exp.cube[0][2], exp.cube[2][2])
    return compute_spectral_intensity(
        M_factor, pks, omega, exp.SE_gen(), fermi, grid_shape
    )


def spectral_maps_from_experiment(
    exp, device: str | torch.device = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Raw + broadened intensity maps; CUDA/MPS when available."""
    from grizzly.utils import to_numpy

    I = build_raw_I_from_experiment(exp, device=device)
    kyg, kxg, wg = chinook_gaussian_sigmas(exp)
    Ig = gaussian_convolution_3d(I, kyg, kxg, wg)
    return to_numpy(I), to_numpy(Ig)


def chinook_gaussian_sigmas(exp):
    """Pixel sigmas (kyg, kxg, wg) matching chinook spectral()."""
    kxg = (
        exp.cube[0][2] * exp.dk / (exp.cube[0][1] - exp.cube[0][0])
        if abs(exp.cube[0][1] - exp.cube[0][0]) > 0
        else 0.0
    )
    kyg = (
        exp.cube[1][2] * exp.dk / (exp.cube[1][1] - exp.cube[1][0])
        if abs(exp.cube[1][1] - exp.cube[1][0]) > 0
        else 0.0
    )
    wg = (
        exp.cube[2][2] * exp.dE / (exp.cube[2][1] - exp.cube[2][0])
        if abs(exp.cube[2][1] - exp.cube[2][0]) > 0
        else 0.0
    )
    return kyg, kxg, wg
