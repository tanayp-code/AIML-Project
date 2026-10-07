#!/usr/bin/env python3
"""
CL-Drive Dataset Preprocessing & Windowing Script
=================================================
Processes raw CL-Drive physiological recordings (ECG, EDA, EEG, Gaze, Labels)
and generates synchronized 10-second uniform 100 Hz sequence tensors.

Input:
  Extracted directory from CL-Drive (containing ECG/, EDA/, EEG/, Gaze/, Labels/)

Output:
  data/cldrive_windowed_100hz.npz  (12-channel synchronized sequence archive)
  data/window_metadata.csv         (per-window labels and participant metadata)

Usage:
  python preprocess_dataset.py --raw_dir /path/to/extracted/cldrive
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Default Sensor Configuration
# ---------------------------------------------------------------------------
WINDOW_DURATION_SEC = 10.0   # 10s windows aligned with label intervals
TARGET_FS_HZ = 100           # Resample all modalities to uniform 100 Hz
NUM_TIME_STEPS = int(WINDOW_DURATION_SEC * TARGET_FS_HZ)  # 1,000 steps

CHANNELS = {
    'ECG': ['ECG LL-RA CAL', 'ECG LA-RA CAL', 'ECG Vx-RL CAL'],
    'EDA': ['GSR Conductance CAL', 'GSR Resistance CAL'],
    'EEG': ['TP9', 'AF7', 'AF8', 'TP10'],
    'Gaze': ['ET_PupilLeft', 'ET_PupilRight', 'Gaze Velocity'],
}

ALL_CHANNEL_NAMES = (
    CHANNELS['ECG'] + CHANNELS['EDA'] + CHANNELS['EEG'] + CHANNELS['Gaze']
)


def find_dataset_root(raw_dir: str) -> str:
    """
    Recursively searches for the directory containing 'ECG', 'EDA', 'EEG', 'Gaze', 'Labels'.
    Handles cases where user extracted zip/tar into nested subdirectories.
    """
    candidates = [raw_dir, os.path.join(raw_dir, 'CL-Drive'), os.path.join(raw_dir, 'raw')]
    for path in candidates:
        if os.path.isdir(path):
            subdirs = set(os.listdir(path))
            if {'ECG', 'EDA', 'Labels'}.issubset(subdirs):
                return os.path.abspath(path)

    # Search up to 2 levels deep
    for root, dirs, _ in os.walk(raw_dir):
        if {'ECG', 'EDA', 'Labels'}.issubset(set(dirs)):
            return os.path.abspath(root)

    raise FileNotFoundError(
        f"Could not find valid CL-Drive modalities (ECG/, EDA/, Labels/) in '{raw_dir}'.\n"
        f"Please verify that the dataset zip/rar files were extracted into this directory."
    )


def load_trial_signals(base_dir: str, participant: str, level: int) -> dict:
    """Load raw continuous signals for a single trial across all 4 modalities."""
    modalities_data = {}
    p_str = str(participant)

    files = {
        'ECG': os.path.join(base_dir, 'ECG', p_str, f'ecg_data_level_{level}.csv'),
        'EDA': os.path.join(base_dir, 'EDA', p_str, f'eda_data_level_{level}.csv'),
        'EEG': os.path.join(base_dir, 'EEG', p_str, f'eeg_data_level_{level}.csv'),
        'Gaze': os.path.join(base_dir, 'Gaze', p_str, f'gaze_data_level_{level}.csv'),
    }

    for mod, path in files.items():
        if not os.path.isfile(path):
            return None
        try:
            df = pd.read_csv(path)
            ts_col = [c for c in df.columns if 'time' in c.lower()]
            if not ts_col:
                return None
            df = df.rename(columns={ts_col[0]: 'Timestamp'})
            modalities_data[mod] = df
        except Exception:
            return None

    return modalities_data


def extract_window_signals(modalities_data: dict, t_start: float, rel_time_end: float) -> np.ndarray:
    """
    Extract and resample a 10-second multi-channel window to uniform 100 Hz grid.
    Returns ndarray of shape (12, 1000).
    """
    w_start = t_start + (rel_time_end - WINDOW_DURATION_SEC)
    w_end = t_start + rel_time_end
    grid = np.linspace(w_start, w_end, NUM_TIME_STEPS)

    channel_arrays = []

    for mod, col_list in CHANNELS.items():
        df_mod = modalities_data[mod]
        ts = df_mod['Timestamp'].values

        for col in col_list:
            if col in df_mod.columns:
                vals = df_mod[col].values
                valid = ~np.isnan(ts) & ~np.isnan(vals)
                if np.sum(valid) >= 2:
                    interp_v = np.interp(grid, ts[valid], vals[valid])
                else:
                    interp_v = np.zeros(NUM_TIME_STEPS, dtype=np.float32)
            else:
                interp_v = np.zeros(NUM_TIME_STEPS, dtype=np.float32)

            interp_v = np.nan_to_num(interp_v, nan=0.0)
            channel_arrays.append(interp_v)

    return np.stack(channel_arrays, axis=0).astype(np.float32)


def discretize_label(score: float) -> int:
    """Maps continuous 1-9 rating to 3 classes (0: Low, 1: Med, 2: High)."""
    if score < 3.67:
        return 0
    elif score <= 6.33:
        return 1
    else:
        return 2


def preprocess(raw_dir: str, out_dir: str):
    root_dir = find_dataset_root(raw_dir)
    print("=" * 78)
    print("  CL-DRIVE DATASET PREPROCESSING & WINDOWING PIPELINE")
    print("=" * 78)
    print(f"Dataset Root      : {root_dir}")
    print(f"Output Directory  : {os.path.abspath(out_dir)}")
    print(f"Sampling Grid     : {TARGET_FS_HZ} Hz ({WINDOW_DURATION_SEC}s windows -> {NUM_TIME_STEPS} points/window)")
    print(f"Total Channels    : {len(ALL_CHANNEL_NAMES)} channels (ECG: 3, EDA: 2, EEG: 4, Gaze: 3)")

    # Discover participants from Labels/ or ECG/
    labels_dir = os.path.join(root_dir, 'Labels')
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.csv')))
    participants = [os.path.splitext(os.path.basename(f))[0] for f in label_files]

    if not participants:
        # Fallback to ECG directories
        ecg_dir = os.path.join(root_dir, 'ECG')
        participants = sorted([d for d in os.listdir(ecg_dir) if os.path.isdir(os.path.join(ecg_dir, d))])

    print(f"Discovered Cohort : {len(participants)} participants ({', '.join(participants[:5])}...)")
    levels = list(range(1, 10))

    os.makedirs(out_dir, exist_ok=True)
    all_windows = []
    metadata_rows = []

    total_trials = len(participants) * len(levels)
    processed_trials = 0

    print(f"\nProcessing {total_trials} participant × level trials...")

    for p in participants:
        lbl_path = os.path.join(labels_dir, f'{p}.csv')
        if not os.path.isfile(lbl_path):
            continue

        try:
            lbl_df = pd.read_csv(lbl_path)
        except Exception:
            continue

        for lvl in levels:
            processed_trials += 1
            lbl_col = f'lvl_{lvl}'
            if lbl_col not in lbl_df.columns:
                continue

            modalities_data = load_trial_signals(root_dir, p, lvl)
            if modalities_data is None:
                continue

            ecg_ts = modalities_data['ECG']['Timestamp'].dropna()
            if len(ecg_ts) < 100:
                continue
            t_start = ecg_ts.iloc[0]

            for w_idx, row in lbl_df.iterrows():
                rel_time = row['time']
                rating = row[lbl_col]

                if pd.isna(rating) or rating <= 0:
                    continue

                window_data = extract_window_signals(modalities_data, t_start, rel_time)
                all_windows.append(window_data)
                metadata_rows.append({
                    'sample_id': len(metadata_rows),
                    'participant': int(p),
                    'level': int(lvl),
                    'window_idx': int(w_idx),
                    'rel_time_end_sec': float(rel_time),
                    'rating_raw': float(rating),
                    'target_3class': discretize_label(float(rating)),
                })

            if processed_trials % 30 == 0 or processed_trials == total_trials:
                print(f"  [{processed_trials:03d}/{total_trials}] Extracted {len(all_windows)} windows so far...")

    if not all_windows:
        raise RuntimeError("No valid windows could be extracted. Please check the dataset files.")

    # Stack into full tensor: (N, 12, 1000)
    X = np.stack(all_windows, axis=0)
    meta_df = pd.DataFrame(metadata_rows)

    y_raw = meta_df['rating_raw'].values.astype(np.float32)
    y_3class = meta_df['target_3class'].values.astype(np.int64)
    p_arr = meta_df['participant'].values.astype(np.int64)
    lvl_arr = meta_df['level'].values.astype(np.int64)

    print("\n" + "=" * 78)
    print("  EXTRACTION COMPLETE")
    print("=" * 78)
    print(f"Total Windows Extracted : {X.shape[0]} windows")
    print(f"Tensor Dimensions (X)   : {X.shape} (N, Channels=12, T=1000)")
    print(f"Memory Footprint        : {X.nbytes / (1024 * 1024):.1f} MB")
    print(f"Unique Drivers Processed: {len(np.unique(p_arr))} drivers")

    # Save output npz
    out_npz = os.path.join(out_dir, 'cldrive_windowed_100hz.npz')
    np.savez_compressed(
        out_npz,
        X=X,
        y_raw=y_raw,
        y_3class=y_3class,
        participant=p_arr,
        level=lvl_arr,
        channel_names=ALL_CHANNEL_NAMES,
    )
    print(f"\nSaved tensor archive -> {os.path.abspath(out_npz)}")

    # Save metadata CSV
    out_csv = os.path.join(out_dir, 'window_metadata.csv')
    meta_df.to_csv(out_csv, index=False)
    print(f"Saved metadata table -> {os.path.abspath(out_csv)}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CL-Drive Dataset into 100 Hz Windows")
    parser.add_argument('--raw_dir', type=str, required=True,
                        help="Path to directory where CL-Drive was extracted (must contain ECG/, EDA/, etc.)")
    parser.add_argument('--out_dir', type=str, default='data',
                        help="Output directory for preprocessed archive (default: 'data')")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess(args.raw_dir, args.out_dir)
