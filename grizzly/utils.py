import torch
import numpy as np

def get_device(device_str='auto'):
    """Select PyTorch device: CUDA > MPS > CPU when ``device_str=='auto'``."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)

def to_torch(x, device, dtype=torch.complex128):
    """Convert array or scalar to ``torch.Tensor`` on ``device`` with ``dtype``."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def to_numpy(t):
    """Convert ``torch.Tensor`` to ``numpy.ndarray`` (detach, CPU)."""
    if isinstance(t, np.ndarray):
        return t
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.array(t)
