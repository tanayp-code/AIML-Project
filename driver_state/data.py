"""
Loading the preprocessed archives as calibrated training sets or as replay streams (numpy only).

A "segment" is one continuous recording that is calibrated on its own:
  cldrive : one segment per driver (levels played in order, calibrated once at the start)
  drozy   : one segment per (subject, session)
"""

import numpy as np

from .config import DATASETS, SAMPLE_RATE, WINDOW_SAMPLES
from .calibration import fit_calibration, apply_calibration


def load_archive(key: str, npz_path: str) -> dict:
    raw = np.load(npz_path, allow_pickle=True)
    X = raw['X'].astype(np.float32)
    participant = raw['participant'].astype(np.int64)
    level = raw['level'].astype(np.int64)

    if key == 'cldrive':
        session = np.zeros_like(participant)
        # Play levels in ascending order; stable sort keeps the in-level window order
        order_key = level
        channel_names = [str(c) for c in raw['channel_names']]
    else:
        session = raw['session'].astype(np.int64)
        order_key = raw['window_idx'].astype(np.int64)
        channel_names = [str(c) for c in raw['channels']]

    # 3-class labels: CL-Drive Low 1-3 / Medium 4-6 / High 7-9; DROZY uses the archive's KSS mapping
    if key == 'cldrive':
        y3 = np.where(level <= 3, 0, np.where(level <= 6, 1, 2)).astype(np.int64)
    else:
        y3 = raw['y_3class'].astype(np.int64)

    return {
        'X': X, 'participant': participant, 'session': session,
        'level': level, 'y3': y3, 'order_key': order_key,
        'channel_names': channel_names,
    }


def segment_indices(arc: dict, subject: int, session: int) -> np.ndarray:
    """Window indices of one segment in playback order."""
    idx = np.where((arc['participant'] == subject) & (arc['session'] == session))[0]
    return idx[np.argsort(arc['order_key'][idx], kind='stable')]


def list_segments(arc: dict, subjects=None):
    pairs = sorted(set(zip(arc['participant'].tolist(), arc['session'].tolist())))
    if subjects is not None:
        keep = set(int(s) for s in subjects)
        pairs = [p for p in pairs if p[0] in keep]
    return pairs


def warmup_window_count(key: str) -> int:
    return max(1, DATASETS[key]['calibration']['seconds'] * SAMPLE_RATE // WINDOW_SAMPLES)


def calibrated_training_set(key: str, arc: dict, subjects=None):
    """
    Returns (X_calibrated, y3, level, participant) with every segment calibrated on its own
    warm-up windows, exactly as the live pipeline will see it.
    """
    method = DATASETS[key]['calibration']['method']
    n_warm = warmup_window_count(key)
    Xs, y3s, lvls, parts = [], [], [], []

    for subject, session in list_segments(arc, subjects):
        idx = segment_indices(arc, subject, session)
        warm_idx = idx[:n_warm]
        # (n_warm, C, 1000) -> (C, n_warm * 1000) in time order
        warmup = np.concatenate(list(arc['X'][warm_idx]), axis=1)
        params = fit_calibration(method, warmup)
        Xs.append(apply_calibration(params, arc['X'][idx]))
        y3s.append(arc['y3'][idx])
        lvls.append(arc['level'][idx])
        parts.append(arc['participant'][idx])

    return (np.concatenate(Xs), np.concatenate(y3s),
            np.concatenate(lvls), np.concatenate(parts))


def replay_segments(arc: dict, subject: int):
    """
    Continuous raw streams for one subject: list of
    {'session', 'X': (C, T) float32, 'levels': per-window level, 'y3': per-window class}.
    """
    segs = []
    for subj, session in list_segments(arc, [subject]):
        idx = segment_indices(arc, subj, session)
        segs.append({
            'session': int(session),
            'X': np.ascontiguousarray(np.concatenate(list(arc['X'][idx]), axis=1)),
            'levels': arc['level'][idx].copy(),
            'y3': arc['y3'][idx].copy(),
        })
    return segs
