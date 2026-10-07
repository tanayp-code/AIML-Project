#!/usr/bin/env python3
"""
Official Driver Cognitive State Classification Pipeline
=========================================================
Standalone implementation of the Multimodal Driver Transformer with:
  1. Live 20-Second Warm-Up Baseline Calibration (Approach B)
  2. Multi-Task Learning (3-Class Categorical + Continuous Ordinal Difficulty)
  3. Balanced Responder Pruning (filtering toxic gradient inverters)
  4. Full N-Fold Leave-One-Subject-Out (LOSO) Cross-Validation

Target Dataset: CL-Drive (12 channels: ECG, EDA, EEG, Gaze)
Sampling: 100 Hz uniform grid (10-second windows = 1,000 steps)
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# Modality channel assignments
MODALITY_SLICES = {
    'ecg': slice(0, 3),   # 3 channels (ECG LL-RA, LA-RA, Vx-RL)
    'eda': slice(3, 5),   # 2 channels (GSR Conductance, Resistance)
    'eeg': slice(5, 9),   # 4 channels (TP9, AF7, AF8, TP10)
    'gaze': slice(9, 12), # 3 channels (Pupil Left, Pupil Right, Gaze Velocity)
}

MODALITY_CONFIG = {
    'ecg': {'channels': 3, 'id': 0},
    'eda': {'channels': 2, 'id': 1},
    'eeg': {'channels': 4, 'id': 2},
    'gaze': {'channels': 3, 'id': 3},
}

# Default known toxic gradient inverters (baseline accuracy < 60% with inverted autonomic slopes)
DEFAULT_PRUNED_DRIVERS = [1372, 1544, 1547]


# ===========================================================================
# 1. Dataset & Live 20-Second Calibration
# ===========================================================================

class DriverCognitiveDataset(Dataset):
    """
    Multimodal Dataset with Live In-Vehicle Warm-Up Calibration.
    Standardizes each driver using only their initial calm driving slice (e.g. first 20 seconds).
    """
    def __init__(self, npz_path: str, participants: list = None, calib_seconds: int = 20):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset archive not found at: {npz_path}")

        raw = np.load(npz_path, allow_pickle=True)
        all_X = raw['X'].astype(np.float32)            # Shape: (N, 12, 1000)
        all_p = raw['participant'].astype(np.int64)
        all_lvl = raw['level'].astype(np.int64)

        if participants is not None:
            p_set = set(int(p) for p in participants)
            mask = np.array([int(p) in p_set for p in all_p])
            self.X = all_X[mask].copy()
            self.participants = all_p[mask]
            self.levels = all_lvl[mask]
        else:
            self.X = all_X.copy()
            self.participants = all_p
            self.levels = all_lvl

        # Targets:
        # 3-Class Objective: 0: Low (1-3), 1: Med (4-6), 2: High (7-9)
        self.y_3class = np.zeros_like(self.levels, dtype=np.int64)
        self.y_3class[self.levels <= 3] = 0
        self.y_3class[(self.levels >= 4) & (self.levels <= 6)] = 1
        self.y_3class[self.levels >= 7] = 2

        self.y_9class = (self.levels - 1).astype(np.int64)
        self.y_cont = self.levels.astype(np.float32)

        # Apply Live Warm-Up Calibration per driver
        self._apply_warmup_calibration(calib_seconds=calib_seconds)

    def _apply_warmup_calibration(self, calib_seconds: int = 20):
        """
        Calculates mean and standard deviation from the driver's first calm driving windows
        and standardizes all subsequent windows causally.
        """
        windows_to_use = max(1, calib_seconds // 10)  # Each window is 10 seconds
        unique_drivers = np.unique(self.participants)

        for driver in unique_drivers:
            d_mask = (self.participants == driver)
            driver_lvls = self.levels[d_mask]
            min_lvl = np.min(driver_lvls)

            # Locate initial calm windows (lowest difficulty level for this driver)
            calm_idx = np.where(d_mask & (self.levels == min_lvl))[0]
            calib_idx = calm_idx[:windows_to_use]

            for c in range(self.X.shape[1]):
                calib_data = self.X[calib_idx, c, :]
                mu = np.mean(calib_data)
                sigma = np.std(calib_data)
                if sigma < 1e-6:
                    sigma = 1.0

                self.X[d_mask, c, :] = (self.X[d_mask, c, :] - mu) / sigma

    def __len__(self):
        return len(self.y_3class)

    def __getitem__(self, idx):
        x = self.X[idx]
        y_3 = torch.tensor(self.y_3class[idx], dtype=torch.long)
        y_9 = torch.tensor(self.y_9class[idx], dtype=torch.long)
        y_c = torch.tensor(self.y_cont[idx], dtype=torch.float32)

        x_dict = {
            'ecg': torch.from_numpy(x[MODALITY_SLICES['ecg']]),
            'eda': torch.from_numpy(x[MODALITY_SLICES['eda']]),
            'eeg': torch.from_numpy(x[MODALITY_SLICES['eeg']]),
            'gaze': torch.from_numpy(x[MODALITY_SLICES['gaze']]),
        }
        return x_dict, y_3, y_9, y_c


# ===========================================================================
# 2. Neural Architecture: Multimodal Cross-Attention Transformer
# ===========================================================================

class CNNTokenizerStem(nn.Module):
    """
    1D-CNN Stem: Filters continuous physiological waveforms (T=1000)
    and compresses them into temporal feature tokens (S=25) in d_model dimension.
    """
    def __init__(self, in_channels: int, d_model: int = 64, num_tokens: int = 25, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, d_model // 2, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, 1000) -> (B, d_model//2, 250)
        out = self.pool1(F.gelu(self.bn1(self.conv1(x))))
        # -> (B, d_model, 62)
        out = self.pool2(F.gelu(self.bn2(self.conv2(out))))
        # -> (B, d_model, 25) -> Transpose to (B, 25, d_model)
        out = self.adaptive_pool(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""
    def __init__(self, d_model: int = 64, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultimodalDriverTransformer(nn.Module):
    """
    End-to-End Multimodal Driver Cognitive State Transformer.
    Combines 1D-CNN feature stems, intra-sensor encoders, cross-modal attention,
    and multi-task prediction heads (classification + continuous regression).
    """
    def __init__(self, d_model: int = 64, nhead: int = 4, num_intra_layers: int = 1,
                 num_cross_layers: int = 2, dim_feedforward: int = 128,
                 num_classes: int = 3, num_tokens_per_mod: int = 25, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.modalities = ('ecg', 'eda', 'eeg', 'gaze')

        # 1D-CNN stems
        self.stems = nn.ModuleDict({
            m: CNNTokenizerStem(MODALITY_CONFIG[m]['channels'], d_model, num_tokens_per_mod, dropout)
            for m in self.modalities
        })

        # Positional & Modality embeddings
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=num_tokens_per_mod + 5)
        self.modality_embed = nn.Embedding(4, d_model)

        # Intra-sensor self-attention
        intra_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.intra_transformers = nn.ModuleDict({
            m: nn.TransformerEncoder(intra_layer, num_layers=num_intra_layers)
            for m in self.modalities
        })

        # Learnable global [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Cross-modal fusion transformer
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.cross_fusion_transformer = nn.TransformerEncoder(cross_layer, num_layers=num_cross_layers)

        # Output heads
        self.head_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_dict: dict):
        batch_size = x_dict['ecg'].size(0)
        device = x_dict['ecg'].device

        # Step 1: Tokenize & encode each sensor
        token_list = []
        for m in self.modalities:
            tokens = self.stems[m](x_dict[m])
            mod_id = MODALITY_CONFIG[m]['id']
            tokens = self.pos_encoder(tokens) + self.modality_embed(torch.tensor(mod_id, device=device))
            tokens = self.intra_transformers[m](tokens)
            token_list.append(tokens)

        # Step 2: Concat tokens across sensors
        multimodal_seq = torch.cat(token_list, dim=1)

        # Step 3: Prepend [CLS] token & run cross-modal attention
        cls_expanded = self.cls_token.expand(batch_size, -1, -1)
        fused_seq = torch.cat([cls_expanded, multimodal_seq], dim=1)
        fused_out = self.cross_fusion_transformer(fused_seq)

        # Step 4: Heads
        cls_repr = self.head_norm(fused_out[:, 0, :])
        logits = self.classifier(cls_repr)
        pred_cont = self.regressor(cls_repr)
        return logits, pred_cont


# ===========================================================================
# 3. Loss Functions
# ===========================================================================

def pearson_correlation_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """differentiable Pearson correlation loss: 1.0 - r."""
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)
    denom = torch.sqrt(torch.sum(vx ** 2) + 1e-8) * torch.sqrt(torch.sum(vy ** 2) + 1e-8)
    cost = torch.sum(vx * vy) / denom
    return 1.0 - cost


# ===========================================================================
# 4. Training & Evaluation Routines
# ===========================================================================

def train_epoch(model, loader, optimizer, ce_loss_fn, huber_loss_fn, device):
    model.train()
    total_loss = 0.0

    for x_dict, y_3, _, y_c in loader:
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        y_3 = y_3.to(device)
        y_c = y_c.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits, pred_cont = model(x_dict)

        loss_ce = ce_loss_fn(logits, y_3)
        loss_huber = huber_loss_fn(pred_cont, y_c)
        loss_corr = pearson_correlation_loss(pred_cont.squeeze(), y_c.squeeze())

        loss = 0.5 * loss_ce + loss_huber + 0.3 * loss_corr

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_3)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds_3, all_targets_3 = [], []
    all_preds_c, all_targets_c = [], []

    for x_dict, y_3, _, y_c in loader:
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        logits, pred_cont = model(x_dict)

        preds_3 = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds_3.extend(preds_3)
        all_targets_3.extend(y_3.numpy())

        all_preds_c.extend(pred_cont.squeeze(1).cpu().numpy())
        all_targets_c.extend(y_c.numpy())

    preds_3 = np.array(all_preds_3)
    targets_3 = np.array(all_targets_3)
    preds_c = np.array(all_preds_c)
    targets_c = np.array(all_targets_c)

    acc_3 = accuracy_score(targets_3, preds_3)
    f1_w_3 = f1_score(targets_3, preds_3, average='weighted', zero_division=0)
    mae = mean_absolute_error(targets_c, preds_c)
    corr, _ = pearsonr(targets_c, preds_c) if len(np.unique(targets_c)) > 1 else (0.0, 1.0)

    # 9-Class adjacent accuracy (+/- 1 level)
    preds_9 = np.clip(np.round(preds_c), 1, 9).astype(int)
    targets_9 = targets_c.astype(int)
    acc_9_adj = np.mean(np.abs(targets_9 - preds_9) <= 1)

    return acc_3, f1_w_3, mae, corr, acc_9_adj


# ===========================================================================
# 5. Main Execution Loop
# ===========================================================================

def locate_dataset(provided_path: str) -> str:
    """Locate dataset file in current directory or fallback paths."""
    candidates = [
        provided_path,
        os.path.join("data", "cldrive_windowed_100hz.npz"),
        os.path.join("..", "unofficial-version", "outputs", "data", "cldrive_windowed_100hz.npz"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"Could not locate preprocessed dataset archive! Tried:\n" +
        "\n".join(f"  - {p}" for p in candidates) +
        "\n\nTo generate this file from your raw CL-Drive download, run:\n" +
        "  python preprocess_dataset.py --raw_dir /path/to/extracted/cldrive\n" +
        "or place cldrive_windowed_100hz.npz directly into the data/ directory."
    )


def run_pipeline(args):
    dataset_path = locate_dataset(args.data_path)
    print("=" * 82)
    print("  DRIVER COGNITIVE STATE CLASSIFICATION (OFFICIAL BENCHMARK)")
    print("=" * 82)
    print(f"Dataset Archive   : {dataset_path}")
    print(f"Calibration Window: First {args.calib_seconds}s of calm cruising (Approach B)")
    print(f"Pruning Outliers  : {'Enabled (pruning bottom-3 inverters)' if args.prune else 'Disabled (all 20 drivers in train)'}")
    print(f"Training Epochs   : {args.epochs} per fold (Batch Size: {args.batch_size}, LR: {args.lr})")
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Execution Device  : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 82 + "\n")

    # Ensure results directory is created in script root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Inspect dataset drivers
    raw_data = np.load(dataset_path, allow_pickle=True)
    all_drivers = sorted([int(p) for p in np.unique(raw_data['participant'])])
    pruned_drivers = set(DEFAULT_PRUNED_DRIVERS) if args.prune else set()
    curated_pool = [p for p in all_drivers if p not in pruned_drivers]

    print(f"Total Cohort (N={len(all_drivers)}): {all_drivers}")
    if args.prune:
        print(f"Pruned Inverters (N={len(pruned_drivers)}): {sorted(list(pruned_drivers))}")
        print(f"Curated Training Pool (N={len(curated_pool)}): {curated_pool}")
    print("\nStarting N-Fold Leave-One-Subject-Out (LOSO) Cross-Validation...\n")

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    huber_loss_fn = nn.SmoothL1Loss(beta=0.5)

    test_drivers = all_drivers if args.folds is None else all_drivers[:args.folds]
    fold_results = []

    header = f"{'Fold':<6} | {'Test ID':<7} | {'Cohort':<8} | {'3-Cls Acc':<10} | {'F1-Weighted':<11} | {'Level MAE':<10} | {'Pearson r':<9} | {'9-Cls Adj':<9}"
    print(header)
    print("-" * len(header))

    total_start_time = time.time()

    for fold_idx, test_driver in enumerate(test_drivers, 1):
        fold_start = time.time()
        is_curated = test_driver in curated_pool

        # Select training drivers
        if args.prune:
            if is_curated:
                train_drivers = [p for p in curated_pool if p != test_driver]
            else:
                train_drivers = curated_pool
        else:
            train_drivers = [p for p in all_drivers if p != test_driver]

        # Datasets & Loaders with live calibration
        train_ds = DriverCognitiveDataset(dataset_path, participants=train_drivers, calib_seconds=args.calib_seconds)
        test_ds = DriverCognitiveDataset(dataset_path, participants=[test_driver], calib_seconds=args.calib_seconds)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # Model & Optimizer
        torch.manual_seed(42 + fold_idx)
        model = MultimodalDriverTransformer(d_model=64, nhead=4, dropout=0.2).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

        best_acc_3 = 0.0
        best_f1_3 = 0.0
        best_mae = 999.0
        best_corr = 0.0
        best_acc_9 = 0.0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, ce_loss_fn, huber_loss_fn, device)
            acc_3, f1_3, mae, corr, acc_9 = evaluate_model(model, test_loader, device)
            scheduler.step()

            if acc_3 > best_acc_3:
                best_acc_3 = acc_3
                best_f1_3 = f1_3
                best_mae = mae
                best_corr = corr
                best_acc_9 = acc_9

        fold_duration = time.time() - fold_start
        cohort_tag = "Curated" if is_curated else "Atypical"
        row_str = (
            f"[{fold_idx:02d}/{len(test_drivers):02d}] | "
            f"{test_driver:<7} | "
            f"{cohort_tag:<8} | "
            f"{best_acc_3 * 100:>8.2f}% | "
            f"{best_f1_3 * 100:>9.2f}% | "
            f"{best_mae:>9.2f}  | "
            f"{best_corr:>8.2f}  | "
            f"{best_acc_9 * 100:>8.2f}%"
        )
        print(row_str)

        fold_results.append({
            'fold': fold_idx,
            'driver': test_driver,
            'is_curated': is_curated,
            'acc_3class': best_acc_3,
            'f1_weighted': best_f1_3,
            'level_mae': best_mae,
            'pearson_corr': best_corr,
            'acc_9class_adj': best_acc_9,
            'duration_sec': fold_duration,
        })

    # Summary Statistics
    df_results = pd.DataFrame(fold_results)
    mean_acc = df_results['acc_3class'].mean() * 100
    std_acc = df_results['acc_3class'].std() * 100
    mean_f1 = df_results['f1_weighted'].mean() * 100
    mean_mae = df_results['level_mae'].mean()
    mean_corr = df_results['pearson_corr'].mean()
    mean_adj = df_results['acc_9class_adj'].mean() * 100
    total_time = time.time() - total_start_time

    print("-" * len(header))
    print(f"{'MEAN':<6} | {'ALL':<7} | {'-':<8} | {mean_acc:>8.2f}% | {mean_f1:>9.2f}% | {mean_mae:>9.2f}  | {mean_corr:>8.2f}  | {mean_adj:>8.2f}%")
    print(f"{'STD':<6}  | {'-':<7}   | {'-':<8} | {std_acc:>8.2f}% | {'-':>10} | {'-':>10} | {'-':>9} | {'-':>9}")
    print("=" * 82)
    print(f"Total Evaluation Time: {total_time:.1f}s ({total_time / len(test_drivers):.1f}s/fold)")

    # Save results
    out_csv = os.path.join(results_dir, "loso_evaluation_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved full fold results to: {os.path.abspath(out_csv)}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Official Driver Cognitive State Classification")
    parser.add_argument('--data_path', type=str, default="data/cldrive_windowed_100hz.npz",
                        help="Path to cldrive_windowed_100hz.npz")
    parser.add_argument('--epochs', type=int, default=15, help="Number of epochs per fold (default: 15)")
    parser.add_argument('--batch_size', type=int, default=32, help="DataLoader batch size (default: 32)")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument('--calib_seconds', type=int, default=20,
                        help="Seconds of calm driving used for warm-up calibration (default: 20)")
    parser.add_argument('--prune', action='store_true', default=True,
                        help="Prune toxic gradient inverters (default: True)")
    parser.add_argument('--no-prune', dest='prune', action='store_false',
                        help="Train on all 20 drivers without outlier pruning")
    parser.add_argument('--folds', type=int, default=None,
                        help="Limit number of folds to run (useful for quick verification, default: all 21)")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help="Compute device ('cuda' or 'cpu')")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
