"""
Dataset and model specifications for the two supported models (numpy-free, torch-free).

  cldrive : CL-Drive cognitive load model (ECG, EDA, EEG, Gaze; 12 channels; 21 drivers)
  drozy   : DROZY drowsiness model (EEG, ECG, EOG, EMG, Face; 15 channels; 14 subjects)
"""

import os

OFFICIAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(OFFICIAL_ROOT)
WEIGHTS_DIR = os.path.join(OFFICIAL_ROOT, "weights")

SAMPLE_RATE = 100      # Hz, every archive is resampled to this grid
WINDOW_SAMPLES = 1000  # 10-second windows

DATASETS = {
    'cldrive': {
        'title': "CL-Drive cognitive load",
        'npz_candidates': [
            os.path.join(OFFICIAL_ROOT, "data", "cldrive_windowed_100hz.npz"),
            os.path.join(PROJECT_ROOT, "cldrive-unofficial", "outputs", "data", "cldrive_windowed_100hz.npz"),
        ],
        # name -> (first channel, last channel + 1, modality embedding id)
        'modalities': {
            'ecg': (0, 3, 0),
            'eda': (3, 5, 1),
            'eeg': (5, 9, 2),
            'gaze': (9, 12, 3),
        },
        'modality_labels': {
            'ecg': "ECG (heart)", 'eda': "EDA (skin)", 'eeg': "EEG (brain)", 'gaze': "Gaze (eye tracker)",
        },
        'num_channels': 12,
        'class_names': ["Low", "Medium", "High"],
        'level_name': "difficulty level",
        'subject_name': "driver",
        # Warm-up calibration: z-score every channel with the first 20 s of calm driving
        'calibration': {'method': 'zscore', 'seconds': 20},
        'train': {'epochs': 15, 'lr': 3e-4, 'batch_size': 32},
        'weights_file': "cldrive_final.pt",
    },
    'drozy': {
        'title': "DROZY drowsiness",
        'npz_candidates': [
            os.path.join(OFFICIAL_ROOT, "data", "drozy_windowed_15ch_100hz.npz"),
            os.path.join(PROJECT_ROOT, "drozy-unofficial", "outputs", "data", "drozy_windowed_15ch_100hz.npz"),
        ],
        'modalities': {
            'eeg': (0, 5, 0),
            'ecg': (5, 6, 1),
            'eog': (6, 8, 2),
            'emg': (8, 9, 3),
            'face': (9, 15, 4),
        },
        'modality_labels': {
            'eeg': "EEG (brain)", 'ecg': "ECG (heart)", 'eog': "EOG (eye movement)",
            'emg': "EMG (muscle)", 'face': "Face (camera)",
        },
        'num_channels': 15,
        'class_names': ["Alert", "Moderate", "Drowsy"],
        'level_name': "KSS sleepiness",
        'subject_name': "subject",
        # Robust warm-up calibration on the first 60 s of each session (median/MAD for
        # biosignals, open-eye ratio for EAR, median offset for mouth and head pose)
        'calibration': {'method': 'robust_drozy', 'seconds': 60},
        'train': {'epochs': 10, 'lr': 5e-4, 'batch_size': 32},
        'weights_file': "drozy_final.pt",
    },
}

# Architecture shared by both models (hybrid CNN tokenizer + cross-modal transformer)
MODEL_HPARAMS = {
    'd_model': 64,
    'nhead': 4,
    'intra_layers': 1,
    'cross_layers': 2,
    'dim_feedforward': 128,
    'num_tokens': 25,
    'num_classes': 3,
    'dropout': 0.2,
}


def weights_path(key: str) -> str:
    return os.path.join(WEIGHTS_DIR, DATASETS[key]['weights_file'])


def locate_npz(key: str, override: str = None) -> str:
    candidates = ([override] if override else []) + DATASETS[key]['npz_candidates']
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"Could not find the preprocessed {DATASETS[key]['title']} archive. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nPass --data_path to point at it."
    )
