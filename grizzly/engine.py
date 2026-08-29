import torch
import numpy as np
import psutil
from grizzly.utils import to_torch, get_device, to_numpy
from grizzly.ylm_torch import batched_Y

def prepare_static_tensors(experiment, device='cpu'):
    """Extract state-independent chinook experiment arrays as PyTorch tensors.

    Parameters
    ----------
    experiment : chinook ARPES experiment
        Must have datacube() already run (prefactors, Ev, pks, etc.).
    device : str or torch.device
        Target device for tensors.

    Returns
    -------
    dict
        Keys: prefactors, proj_arr, Gbasis, orbital_pointers, radint_pointers,
        Largs, Margs, pks, Ev, ph, th.
    """
    dev = get_device(device)
    
    return {
        'prefactors': to_torch(experiment.prefactors, dev, dtype=torch.float64),
        'proj_arr': to_torch(experiment.proj_arr, dev, dtype=torch.complex128),
        'Gbasis': to_torch(experiment.Gbasis, dev, dtype=torch.float64),
        'orbital_pointers': to_torch(experiment.orbital_pointers, dev, dtype=torch.int64),
        'radint_pointers': to_torch(experiment.radint_pointers, dev, dtype=torch.int64),
        'Largs': to_torch(experiment.Largs, dev, dtype=torch.int64),
        'Margs': to_torch(experiment.Margs, dev, dtype=torch.int64),
        'pks': to_torch(experiment.pks, dev, dtype=torch.float64),
        'Ev': to_torch(experiment.Ev, dev, dtype=torch.complex128),
        'ph': to_torch(experiment.ph, dev, dtype=torch.float64),
        'th': to_torch(experiment.th, dev, dtype=torch.float64)
    }

def prepare_batch_inputs(batch_indices, static_dict, experiment):
    """
    Gathers per-state inputs from the static tensors for a batch.
    Evaluates radial integrals on CPU and moves them to device.
    """
    indices = to_torch(batch_indices, static_dict['pks'].device, dtype=torch.int64)
    
    pks_batch = static_dict['pks'][indices]
    
    nstates = len(experiment.TB.basis)
    k_indices = torch.div(pks_batch[:, 0].to(torch.int64), nstates, rounding_mode='floor')
    band_indices = pks_batch[:, 0].to(torch.int64) % nstates
    
    Ev_batch = static_dict['Ev'][k_indices, :, band_indices]
    theta_batch = static_dict['th'][indices]
    phi_batch = static_dict['ph'][k_indices]
    
    # Radial integral evaluation on CPU
    energies_batch = to_numpy(pks_batch[:, 3])
    B_eval_batch_np = np.zeros((len(energies_batch), len(experiment.Bfuncs), 2), dtype=np.complex128)
    
    for i, b in enumerate(experiment.Bfuncs):
        B_eval_batch_np[:, i, 0] = b[0](energies_batch)
        B_eval_batch_np[:, i, 1] = b[1](energies_batch)
        
    B_eval_batch = to_torch(B_eval_batch_np, static_dict['pks'].device, dtype=torch.complex128)
    
    return Ev_batch, theta_batch, phi_batch, B_eval_batch


def compute_ylm_batch(theta_batch, phi_batch, Largs, Margs):
    """
    Evaluates spherical harmonics for all states in the batch.
    Returns (B, N_unique_lm, 2, 3)
    """
    B = theta_batch.shape[0]
    N_unique_lm = Largs.shape[0]
    
    L_flat = Largs.reshape(-1)
    M_flat = Margs.reshape(-1)
    
    Y_flat = batched_Y(L_flat, M_flat, theta_batch, phi_batch, device=theta_batch.device) # (B, N_unique_lm * 6)
    
    return Y_flat.reshape(B, N_unique_lm, 2, 3)


def batched_matrix_elements(
    Ev_batch,          # (B, N_orbs) complex128
    B_eval_batch,      # (B, N_unique_rad, 2) complex128
    Ylm_batch,         # (B, N_unique_lm, 2, 3) complex128
    prefactors,        # (N_orbs,) float64
    proj_arr,          # (N_orbs, max_proj) complex128
    Gbasis,            # (N_orbs, max_proj, 2, 3) float64
    orbital_pointers,  # (N_orbs, max_proj) int64
    radint_pointers,   # (N_orbs,) int64
    spin: bool = False
) -> torch.Tensor:
    
    B = Ev_batch.shape[0]
    
    # Step 1: pref = prefactors * Ev * B_eval
    scaled = prefactors.unsqueeze(0) * Ev_batch                 # (B, N_orbs)
    B_indexed = B_eval_batch[:, radint_pointers, :]             # (B, N_orbs, 2)
    pref = scaled.unsqueeze(-1) * B_indexed                     # (B, N_orbs, 2)
    
    # Step 2: Ylm x Gaunt, summed over projections
    Ylm_indexed = Ylm_batch[:, orbital_pointers, :, :]          # (B, N_orbs, max_proj, 2, 3)
    YG = Ylm_indexed * Gbasis.unsqueeze(0)                      # (B, N_orbs, max_proj, 2, 3)
    Gtmp = torch.einsum('boprc, op -> borc', YG, proj_arr)      # (B, N_orbs, 2, 3)
    
    # Step 3: Final contraction
    Mk = torch.zeros((B, 2, 3), dtype=Ev_batch.dtype, device=Ev_batch.device)
    if spin:
        half = Ev_batch.shape[1] // 2
        Mk[:, 0, :] = torch.einsum('bor, borc -> bc', pref[:, :half, :], Gtmp[:, :half, :, :])
        Mk[:, 1, :] = torch.einsum('bor, borc -> bc', pref[:, half:, :], Gtmp[:, half:, :, :])
    else:
        Mk[:, 0, :] = torch.einsum('bor, borc -> bc', pref, Gtmp)
        
    return Mk


def compute_all_Mk(experiment, batch_size: int = 0, device: str = 'auto') -> np.ndarray:
    """Compute photoemission matrix elements for all valid peaks (batched).

    Replaces chinook ``serial_Mk`` with batched PyTorch contractions.
    Radial integrals are evaluated on CPU; Ylm and einsum run on ``device``.

    Parameters
    ----------
    experiment : chinook ARPES experiment
        After ``datacube()``; uses ``experiment.Mk`` shape and ``experiment.pks``.
    batch_size : int
        States per batch. ``0`` auto-tunes from available GPU/RAM (~70% budget).
    device : str
        ``auto``, ``cuda``, ``cpu``, or ``mps`` (float64 may fail on MPS).

    Returns
    -------
    numpy.ndarray
        Complex128 array shape ``(N_peaks, 2, 3)`` — same layout as chinook ``Mk``.
    """
    static_dict = prepare_static_tensors(experiment, device)
    
    th = static_dict['th']
    valid_indices = torch.where(th >= 0)[0]
    
    if len(valid_indices) == 0:
        return experiment.Mk
        
    N_peaks = experiment.Mk.shape[0]
    Mk_out = np.zeros((N_peaks, 2, 3), dtype=np.complex128)
    
    spin = getattr(experiment, 'spin', False)
    if spin:
        from grizzly.future import require_spinless
        require_spinless(experiment, feature="compute_all_Mk")
    
    if batch_size <= 0:
        dev_obj = get_device(device)
        if dev_obj.type == 'cuda':
            available_mem = torch.cuda.mem_get_info()[0]
        else:
            available_mem = psutil.virtual_memory().available
            
        N_orbs = len(experiment.TB.basis)
        max_proj = static_dict['proj_arr'].shape[1]
        bytes_per_state = N_orbs * max_proj * 2 * 3 * 16 * 3 
        batch_size = max(1, int(0.7 * available_mem / max(bytes_per_state, 1)))

    # Process in batches
    for i in range(0, len(valid_indices), batch_size):
        chunk_indices = valid_indices[i : i + batch_size]
        
        Ev_batch, theta_batch, phi_batch, B_eval_batch = prepare_batch_inputs(chunk_indices, static_dict, experiment)
        
        Ylm_batch = compute_ylm_batch(
            theta_batch, 
            phi_batch, 
            static_dict['Largs'], 
            static_dict['Margs']
        )
        
        Mk_batch = batched_matrix_elements(
            Ev_batch, 
            B_eval_batch, 
            Ylm_batch, 
            static_dict['prefactors'], 
            static_dict['proj_arr'], 
            static_dict['Gbasis'], 
            static_dict['orbital_pointers'], 
            static_dict['radint_pointers'], 
            spin=spin
        )
        
        Mk_out[to_numpy(chunk_indices)] = to_numpy(Mk_batch)
        
    return Mk_out
