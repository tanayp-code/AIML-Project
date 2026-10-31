"""
Warm-up calibration shared by training and live inference.

Every method reduces to per-channel parameters:
    x_calibrated = (x - offset) / scale, then clipped to [-5, 5] where clip is True
so the exact same transform is used offline (train_final.py) and online (inference.py).
"""

import numpy as np


def fit_calibration(method: str, warmup: np.ndarray) -> dict:
    """
    warmup: (C, T) array holding the first `seconds` of samples of a drive/session.
    Returns {'offset': (C,), 'scale': (C,), 'clip': (C,) bool}.
    """
    warmup = np.asarray(warmup, dtype=np.float64)
    C = warmup.shape[0]
    offset = np.zeros(C)
    scale = np.ones(C)
    clip = np.zeros(C, dtype=bool)

    if method == 'zscore':
        # CL-Drive: mean / std of the calm warm-up (matches main.py Approach B)
        for c in range(C):
            offset[c] = warmup[c].mean()
            sigma = warmup[c].std()
            scale[c] = sigma if sigma >= 1e-6 else 1.0

    elif method == 'robust_drozy':
        # DROZY biosignals (EEG, ECG, EOG, EMG = channels 0..8): median / MAD, clipped
        for c in range(min(9, C)):
            med = np.median(warmup[c])
            mad = 1.4826 * np.median(np.abs(warmup[c] - med))
            if mad < 1e-5:
                mad = warmup[c].std()
                if mad < 1e-5:
                    mad = 1.0
            offset[c], scale[c], clip[c] = med, mad, True
        if C >= 15:
            # Eye aspect ratio (9, 10): ratio to the open-eye median
            for c in (9, 10):
                open_eye = np.median(warmup[c])
                scale[c] = open_eye if open_eye >= 1e-4 else 0.40
            # Mouth aspect ratio (11) and head pitch / yaw / roll (12..14): median offset
            for c in (11, 12, 13, 14):
                offset[c] = np.median(warmup[c])
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return {'offset': offset.astype(np.float32), 'scale': scale.astype(np.float32), 'clip': clip}


def apply_calibration(params: dict, x: np.ndarray) -> np.ndarray:
    """x: (..., C, T). Returns a new float32 array."""
    off = params['offset'][:, None]
    scl = params['scale'][:, None]
    out = ((x - off) / scl).astype(np.float32)
    if params['clip'].any():
        idx = np.where(params['clip'])[0]
        out[..., idx, :] = np.clip(out[..., idx, :], -5.0, 5.0)
    return out
