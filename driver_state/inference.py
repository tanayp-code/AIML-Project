"""
Inference process: receives sensor packets, keeps synchronised buffers, runs the warm-up
calibration, and every `interval` seconds predicts on the latest 10-second window.

Status messages sent to the display:
    ('ready', info) once the model is loaded
    ('update', state) every interval
    ('done', state) when every sensor has finished and the last window was scored
    ('error', text)
"""

import math
import time
import queue
import signal
import numpy as np

from .config import SAMPLE_RATE, WINDOW_SAMPLES
from .calibration import fit_calibration, apply_calibration

EMA_TAU_SECONDS = 20.0  # smoothing time constant in stream seconds (causal)


def inference_process(weights_path, device_request, modalities, segment_lengths, truth,
                      interval, in_q, out_q, stop_event):
    """
    modalities: dict name -> (first channel, last channel + 1, id)
    segment_lengths: samples per segment; truth: list per segment of (levels, y3) per window
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import torch
        from .device import resolve_device
        from .models import load_checkpoint

        device, device_note = resolve_device(device_request)
        try:
            model, ckpt = load_checkpoint(weights_path, device)
        except RuntimeError as exc:
            if device.type != 'cuda':
                raise
            device, device_note = torch.device('cpu'), f"CPU (GPU failed while loading: {str(exc)[:60]})"
            model, ckpt = load_checkpoint(weights_path, device)

        num_channels = ckpt['num_channels']
        calib = ckpt['calibration']
        calib_samples = calib['seconds'] * SAMPLE_RATE

        def forward(window):
            x = torch.from_numpy(window[None]).to(device)
            with torch.no_grad():
                logits, level = model(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            return torch.softmax(logits, 1)[0].cpu().numpy(), float(level[0].cpu())

        # Warm-up pass (first CUDA call is slow); fall back to CPU if the GPU fails here
        try:
            forward(np.zeros((num_channels, WINDOW_SAMPLES), dtype=np.float32))
        except RuntimeError as exc:
            if device.type != 'cuda':
                raise
            device, device_note = torch.device('cpu'), f"CPU (GPU failed at first run: {str(exc)[:60]})"
            model.to(device)
            forward(np.zeros((num_channels, WINDOW_SAMPLES), dtype=np.float32))

        out_q.put(('ready', {
            'device': device_note,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trained_subjects': ckpt['trained_subjects'],
            'created': ckpt.get('created', '?'),
        }))
    except Exception as exc:
        out_q.put(('error', f"Inference process failed to start: {type(exc).__name__}: {exc}"))
        return

    names = list(modalities)
    seg_buffers = {}                               # segment -> (C, T) array
    filled = {m: {} for m in names}                # modality -> {segment: contiguous samples}
    received = {m: 0 for m in names}
    last_rx = {m: None for m in names}
    finished = set()

    current_seg = 0
    params = None
    ema = None
    last_avail = 0
    last_pred = None
    n_scored, n_agree = 0, 0
    history = []
    latency_ms = None
    seg_offsets = np.concatenate([[0], np.cumsum(segment_lengths)])

    def segment_buffer(seg):
        if seg not in seg_buffers:
            seg_buffers[seg] = np.zeros((num_channels, segment_lengths[seg]), dtype=np.float32)
        return seg_buffers[seg]

    def handle(msg):
        kind = msg[0]
        if kind == 'chunk':
            _, m, seg, start, data = msg
            a, b, _ = modalities[m]
            segment_buffer(seg)[a:b, start:start + data.shape[1]] = data
            filled[m][seg] = start + data.shape[1]
            received[m] += data.shape[1]
            last_rx[m] = time.time()
        elif kind == 'eof':
            finished.add(msg[1])

    next_tick = time.monotonic() + interval
    while not stop_event.is_set():
        # Collect packets until the next tick
        while True:
            timeout = next_tick - time.monotonic()
            if timeout <= 0:
                break
            try:
                handle(in_q.get(timeout=min(timeout, 0.05)))
            except queue.Empty:
                pass
        next_tick += interval
        if next_tick < time.monotonic():           # we fell behind; do not burst
            next_tick = time.monotonic() + interval

        # Advance to the newest segment every sensor has reached
        reached = min((max(filled[m]) if filled[m] else -1) for m in names)
        if reached > current_seg:
            for old in [s for s in seg_buffers if s < reached]:
                del seg_buffers[old]
            current_seg, params, ema, last_avail, last_pred = reached, None, None, 0, None

        avail = min(filled[m].get(current_seg, 0) for m in names)
        all_done = len(finished) == len(names)

        state = {
            'segment': current_seg,
            'num_segments': len(segment_lengths),
            'stream_seconds': (seg_offsets[current_seg] + avail) / SAMPLE_RATE,
            'segment_seconds': avail / SAMPLE_RATE,
            'sensors': {m: {'samples': received[m], 'last_rx': last_rx[m], 'done': m in finished}
                        for m in names},
            'device': device_note,
            'latency_ms': latency_ms,
            'history': history[-60:],
            'scored': n_scored,
            'agree': n_agree,
            'prediction': last_pred,
        }

        if avail == 0:
            state['status'] = 'WAITING FOR SENSORS'
        elif params is None and avail < calib_samples:
            state['status'] = f"CALIBRATING  {avail / SAMPLE_RATE:4.1f} / {calib['seconds']} s warm-up"
        else:
            if params is None:
                warmup = seg_buffers[current_seg][:, :calib_samples]
                params = fit_calibration(calib['method'], warmup)
            state['status'] = 'RUNNING'

            if avail >= WINDOW_SAMPLES and avail > last_avail:
                window = apply_calibration(params, seg_buffers[current_seg][:, avail - WINDOW_SAMPLES:avail])
                t0 = time.perf_counter()
                try:
                    probs, level = forward(window)
                except RuntimeError as exc:
                    if device.type != 'cuda':
                        raise
                    device = torch.device('cpu')
                    device_note = f"CPU (GPU failed during inference: {str(exc)[:50]})"
                    model.to(device)
                    probs, level = forward(window)
                latency_ms = (time.perf_counter() - t0) * 1000

                dt = (avail - last_avail) / SAMPLE_RATE if ema is not None else None
                if ema is None:
                    ema = probs.copy()
                else:
                    alpha = 1.0 - math.exp(-dt / EMA_TAU_SECONDS)
                    ema = alpha * probs + (1 - alpha) * ema
                last_avail = avail

                levels, y3 = truth[current_seg]
                w = min((avail - 1) // WINDOW_SAMPLES, len(levels) - 1)
                pred = {
                    'probs': probs.tolist(), 'smoothed': ema.tolist(),
                    'cls': int(np.argmax(probs)), 'smoothed_cls': int(np.argmax(ema)),
                    'level': level,
                    'true_level': int(levels[w]), 'true_cls': int(y3[w]),
                }
                n_scored += 1
                n_agree += int(pred['smoothed_cls'] == pred['true_cls'])
                history.append(pred['smoothed_cls'])
                last_pred = pred
                state.update({'prediction': pred, 'latency_ms': latency_ms, 'device': device_note,
                              'history': history[-60:], 'scored': n_scored, 'agree': n_agree})

        if all_done and in_q.empty() and avail >= segment_lengths[current_seg] \
                and current_seg == len(segment_lengths) - 1 and avail == last_avail:
            state['status'] = 'REPLAY FINISHED'
            out_q.put(('done', state))
            return
        out_q.put(('update', state))
