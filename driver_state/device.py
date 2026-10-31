"""
Compute device selection with CPU fallback.

'auto' and 'cuda' both probe the GPU with a real allocation and kernel launch, so a broken
driver, a missing CUDA runtime or an out-of-memory GPU falls back to CPU instead of crashing.
"""

import warnings
import torch


def resolve_device(requested: str = 'auto'):
    """Returns (torch.device, human-readable note)."""
    requested = (requested or 'auto').lower()
    if requested == 'cpu':
        return torch.device('cpu'), "CPU (requested)"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            available = torch.cuda.is_available()
        if not available:
            raise RuntimeError("torch.cuda.is_available() is False (no GPU or driver not loaded)")
        probe = torch.ones(8, device='cuda') * 2.0
        torch.cuda.synchronize()
        if float(probe.sum().item()) != 16.0:
            raise RuntimeError("GPU probe returned a wrong result")
        name = torch.cuda.get_device_name(0)
        return torch.device('cuda'), f"CUDA ({name})"
    except Exception as exc:  # driver/runtime problems surface as many exception types
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return torch.device('cpu'), f"CPU (GPU unavailable, fell back: {reason[:90]})"
