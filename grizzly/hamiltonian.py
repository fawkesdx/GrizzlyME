"""
GPU/CPU batched H(k) assembly + diagonalization.

Mirrors chinook.TB_lib.TB_model.solve_H: fill upper-triangle H_ij(k) from
Fourier-summed hoppings, then eigh(..., UPLO='U').
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from .utils import get_device, to_numpy, to_torch


def extract_hopping_data(
    TB_model: Any,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Flatten chinook ``H_me`` hoppings into PyTorch tensors.

    Returns dict with:
      i_indices (Nh,), j_indices (Nh,), R_vectors (Nh, 3), amplitudes (Nh,),
      n_basis (int)
    """
    device = get_device(device) if isinstance(device, str) else device

    i_list, j_list, R_list, amp_list = [], [], [], []
    for me in TB_model.mat_els:
        if getattr(me, "executable", False):
            raise NotImplementedError(
                "Executable / low-energy H_me entries are not supported in "
                "grizzly.hamiltonian; use standard tij hoppings."
            )
        for hop in me.H:
            # hop = [R0, R1, R2, H]
            i_list.append(me.i)
            j_list.append(me.j)
            R_list.append([hop[0], hop[1], hop[2]])
            amp_list.append(complex(hop[3]))

    if not i_list:
        raise ValueError("TB_model.mat_els has no hopping entries")

    return {
        "i_indices": torch.tensor(i_list, device=device, dtype=torch.long),
        "j_indices": torch.tensor(j_list, device=device, dtype=torch.long),
        "R_vectors": to_torch(
            np.asarray(R_list, dtype=np.float64), device, dtype=torch.float64
        ),
        "amplitudes": to_torch(
            np.asarray(amp_list, dtype=np.complex128), device, dtype=torch.complex128
        ),
        "n_basis": int(len(TB_model.basis)),
    }


def _assemble_Hmat(
    kpts: torch.Tensor,
    hopping_data: Dict[str, Any],
) -> torch.Tensor:
    """
    Build H(k) for a block of k-points. Shape (Nk, Nb, Nb), complex128.

    Only (i, j) entries from hoppings are written (chinook upper-triangle
    convention). Lower triangle left zero for eigh(..., UPLO='U').
    """
    device = hopping_data["i_indices"].device
    Nb = hopping_data["n_basis"]
    Nk = kpts.shape[0]
    Nh = hopping_data["i_indices"].shape[0]

    # phases: (Nk, Nh) = exp(i k · R)
    phases = torch.exp(
        1j
        * torch.einsum(
            "ki,hi->kh",
            kpts.to(device=device, dtype=torch.float64),
            hopping_data["R_vectors"],
        )
    )
    contrib = phases * hopping_data["amplitudes"][None, :]  # (Nk, Nh)

    Hmat = torch.zeros((Nk, Nb, Nb), dtype=torch.complex128, device=device)
    k_idx = torch.arange(Nk, device=device)[:, None].expand(Nk, Nh)
    i_idx = hopping_data["i_indices"][None, :].expand(Nk, Nh)
    j_idx = hopping_data["j_indices"][None, :].expand(Nk, Nh)
    Hmat.index_put_((k_idx, i_idx, j_idx), contrib, accumulate=True)
    return Hmat


def build_and_diagonalize(
    kpts: Union[np.ndarray, torch.Tensor],
    hopping_data: Dict[str, Any],
    device: Union[str, torch.device] = "cpu",
    chunk_size: int = 0,
    Eonly: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Vectorized Fourier sum + ``torch.linalg.eigh`` (UPLO='U').

    *args*:
        kpts: (Nk, 3) Cartesian k
        hopping_data: from ``extract_hopping_data``
        chunk_size: if >0, diagonalize in chunks of this many k-points
        Eonly: eigenvalues only (Evec returned as None)

    *return*:
        Eband (Nk, Nb), Evec (Nk, Nb, Nb) or None
    """
    device = get_device(device) if isinstance(device, str) else device
    if isinstance(kpts, np.ndarray):
        k_t = torch.as_tensor(kpts, device=device, dtype=torch.float64)
    else:
        k_t = kpts.to(device=device, dtype=torch.float64)

    Nk = k_t.shape[0]
    Nb = hopping_data["n_basis"]

    # Ensure hopping tensors on same device
    hop = {
        "i_indices": hopping_data["i_indices"].to(device),
        "j_indices": hopping_data["j_indices"].to(device),
        "R_vectors": hopping_data["R_vectors"].to(device),
        "amplitudes": hopping_data["amplitudes"].to(device),
        "n_basis": Nb,
    }

    if chunk_size is None or chunk_size <= 0 or chunk_size >= Nk:
        Hmat = _assemble_Hmat(k_t, hop)
        if Eonly:
            Eband = torch.linalg.eigvalsh(Hmat, UPLO="U")
            return Eband, None
        Eband, Evec = torch.linalg.eigh(Hmat, UPLO="U")
        return Eband, Evec

    Eband = torch.empty((Nk, Nb), dtype=torch.float64, device=device)
    Evec = None if Eonly else torch.empty(
        (Nk, Nb, Nb), dtype=torch.complex128, device=device
    )
    for start in range(0, Nk, chunk_size):
        stop = min(start + chunk_size, Nk)
        Hblk = _assemble_Hmat(k_t[start:stop], hop)
        if Eonly:
            Eband[start:stop] = torch.linalg.eigvalsh(Hblk, UPLO="U")
        else:
            e, v = torch.linalg.eigh(Hblk, UPLO="U")
            Eband[start:stop] = e
            Evec[start:stop] = v
    return Eband, Evec


def solve_H(
    TB_model: Any,
    device: Union[str, torch.device] = "auto",
    chunk_size: int = 0,
    Eonly: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Drop-in-ish replacement for ``TB_model.solve_H`` using batched PyTorch.

    Reads ``TB_model.Kobj.kpts`` and ``TB_model.mat_els``. Does **not** write
    back onto the TB object (caller may assign if desired).
    """
    if TB_model.Kobj is None:
        raise ValueError("TB_model has no Kobj / k-points defined")
    device = get_device(device) if isinstance(device, str) else device
    data = extract_hopping_data(TB_model, device=device)
    kpts = np.asarray(TB_model.Kobj.kpts, dtype=np.float64)
    return build_and_diagonalize(
        kpts, data, device=device, chunk_size=chunk_size, Eonly=Eonly
    )
