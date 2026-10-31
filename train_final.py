#!/usr/bin/env python3
"""
Train the final deployable models on every subject and save their weights.

  python train_final.py                       # both models (CL-Drive on 21 drivers, DROZY on 14 subjects)
  python train_final.py --dataset cldrive     # one model only
  python train_final.py --device cpu          # force CPU (default 'auto' falls back to CPU if the GPU fails)

No pruning and no held-out subject: these are the models used by live_demo.py.
Accuracy figures for unseen subjects come from the LOSO benchmark (main.py), not from here.
Weights are written to weights/cldrive_final.pt and weights/drozy_final.pt.
"""

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

from driver_state.config import DATASETS, MODEL_HPARAMS, WEIGHTS_DIR, locate_npz, weights_path
from driver_state.data import load_archive, calibrated_training_set
from driver_state.models import MultimodalDriverTransformer
from driver_state.device import resolve_device


def pearson_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    vx = y_pred - y_pred.mean()
    vy = y_true - y_true.mean()
    denom = torch.sqrt((vx ** 2).sum() + 1e-8) * torch.sqrt((vy ** 2).sum() + 1e-8)
    return 1.0 - (vx * vy).sum() / denom


def train_model(key, X, y3, level, device, epochs, lr, batch_size, seed):
    spec = DATASETS[key]
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y3), torch.from_numpy(level.astype(np.float32)))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=gen)

    model = MultimodalDriverTransformer(spec['modalities'], **MODEL_HPARAMS).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    huber_loss = nn.SmoothL1Loss(beta=0.5)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, correct, n = 0.0, 0, 0
        for xb, yb, lb in loader:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            optimizer.zero_grad()
            logits, pred_lvl = model(xb)
            loss = 0.5 * ce_loss(logits, yb) + huber_loss(pred_lvl, lb) + 0.3 * pearson_loss(pred_lvl, lb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        scheduler.step()
        history.append({'epoch': epoch, 'loss': tot_loss / n, 'train_acc': correct / n})
        print(f"  epoch {epoch:2d}/{epochs}  loss {tot_loss / n:.4f}  train acc {100 * correct / n:5.1f}%  "
              f"({time.time() - t0:.1f}s)")
    return model, history


@torch.no_grad()
def training_fit(model, X, y3, level, device, batch_size=128):
    """Accuracy on the training data itself (sanity check only, not generalisation)."""
    model.eval()
    preds, lvls = [], []
    for i in range(0, len(X), batch_size):
        logits, pred_lvl = model(torch.from_numpy(X[i:i + batch_size]).to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        lvls.append(pred_lvl.cpu().numpy())
    preds, lvls = np.concatenate(preds), np.concatenate(lvls)
    return {'acc_3class': float((preds == y3).mean()), 'level_mae': float(np.abs(lvls - level).mean())}


def run(key, args):
    spec = DATASETS[key]
    npz_path = locate_npz(key, args.data_path if args.dataset != 'all' else None)
    epochs = args.epochs or spec['train']['epochs']
    lr = spec['train']['lr']
    batch_size = spec['train']['batch_size']

    print("=" * 78)
    print(f"  FINAL MODEL: {spec['title']}")
    print("=" * 78)
    print(f"Archive     : {npz_path}")

    arc = load_archive(key, npz_path)
    subjects = sorted(int(s) for s in np.unique(arc['participant']))
    X, y3, level, _ = calibrated_training_set(key, arc, subjects)
    cal = spec['calibration']
    print(f"Subjects    : {len(subjects)} ({', '.join(map(str, subjects))}), no pruning, none held out")
    print(f"Windows     : {len(X)} x {X.shape[1]} channels x {X.shape[2]} samples")
    print(f"Calibration : {cal['method']} on the first {cal['seconds']} s of each "
          f"{'drive' if key == 'cldrive' else 'session'}")
    print(f"Training    : {epochs} epochs, lr {lr}, batch {batch_size}, seed {args.seed}")

    device, note = resolve_device(args.device)
    print(f"Device      : {note}\n")

    try:
        model, history = train_model(key, X, y3, level, device, epochs, lr, batch_size, args.seed)
    except RuntimeError as exc:
        if device.type != 'cuda':
            raise
        print(f"\nGPU error during training ({str(exc).splitlines()[0][:100]}). Restarting on CPU.\n")
        device = torch.device('cpu')
        model, history = train_model(key, X, y3, level, device, epochs, lr, batch_size, args.seed)

    fit = training_fit(model, X, y3, level, device)
    print(f"\nFit on training data (not a test score): 3-class acc {100 * fit['acc_3class']:.1f}%, "
          f"level MAE {fit['level_mae']:.2f}")

    checkpoint = {
        'format_version': 1,
        'dataset': key,
        'title': spec['title'],
        'model_config': {
            'modalities': {m: list(v) for m, v in spec['modalities'].items()},
            'hparams': dict(MODEL_HPARAMS),
        },
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'num_channels': spec['num_channels'],
        'channel_names': arc['channel_names'],
        'class_names': spec['class_names'],
        'level_name': spec['level_name'],
        'calibration': dict(cal),
        'trained_subjects': subjects,
        'num_windows': int(len(X)),
        'epochs': epochs,
        'lr': lr,
        'seed': args.seed,
        'training_history': history,
        'training_fit': fit,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'torch_version': str(torch.__version__),
    }
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    out = weights_path(key)
    torch.save(checkpoint, out)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved       : {out} ({n_params:,} parameters, {os.path.getsize(out) / 1e6:.1f} MB)\n")


def parse_args():
    p = argparse.ArgumentParser(description="Train final models on all subjects and save weights")
    p.add_argument('--dataset', choices=['cldrive', 'drozy', 'all'], default='all')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                   help="'auto'/'cuda' use the GPU when it works, otherwise fall back to CPU")
    p.add_argument('--epochs', type=int, default=None, help="Override epochs (default: 15 CL-Drive, 10 DROZY)")
    p.add_argument('--data_path', type=str, default=None, help="Archive path (single-dataset runs only)")
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    keys = ['cldrive', 'drozy'] if args.dataset == 'all' else [args.dataset]
    for k in keys:
        run(k, args)
