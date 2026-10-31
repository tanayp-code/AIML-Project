#!/usr/bin/env python3
"""
Live multi-process demo of the trained models, in the terminal.

Processes started:
  - one sensor simulator per modality (ECG, EDA, EEG, Gaze for CL-Drive;
    EEG, ECG, EOG, EMG, Face for DROZY), each streaming timestamped packets
  - one inference process that buffers the streams, runs the warm-up calibration
    and predicts on the latest 10-second window every --interval seconds
  - this launcher, which draws the live dashboard

  python live_demo.py                              # asks which model, device, subject, speed
  python live_demo.py --model cldrive --device auto --subject 1030 --speed 5
  python live_demo.py --model drozy --device cpu --subject 4 --interval 2

Requires the weights from train_final.py. The simulators replay recorded sessions; the
final models were trained on every subject, so the replayed subject was seen in training.
"""

import os
import sys
import time
import queue
import argparse
import multiprocessing as mp

import numpy as np

from driver_state.config import DATASETS, locate_npz, weights_path
from driver_state.data import load_archive, replay_segments
from driver_state.display import render, Screen

CHUNK_SAMPLES = 20  # 0.2 s packets at 100 Hz


def ask(prompt, choices=None, default=None, cast=str):
    """Interactive question with a default; choices is a list of accepted answers."""
    hint = f" [{default}]" if default is not None else ""
    while True:
        answer = input(f"{prompt}{hint}: ").strip()
        if not answer and default is not None:
            answer = str(default)
        try:
            value = cast(answer)
        except ValueError:
            print("  Invalid value, try again.")
            continue
        if choices is None or value in choices:
            return value
        print(f"  Choose one of: {', '.join(map(str, choices))}")


def parse_args():
    p = argparse.ArgumentParser(description="Live terminal demo: simulated sensors -> model -> dashboard")
    p.add_argument('--model', choices=['cldrive', 'drozy'], help="Which trained model to run")
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'],
                   help="'auto'/'cuda' use the GPU and fall back to CPU if the driver fails")
    p.add_argument('--subject', type=int, help="Driver / subject whose recording is replayed")
    p.add_argument('--speed', type=float, help="Replay speed (1 = real time)")
    p.add_argument('--interval', type=float, default=1.0, help="Seconds between dashboard updates (default 1)")
    p.add_argument('--weights', type=str, help="Checkpoint path (default weights/<model>_final.pt)")
    p.add_argument('--data_path', type=str, help="Preprocessed archive path")
    p.add_argument('--plain', action='store_true', help="Print one line per update instead of a dashboard")
    return p.parse_args()


def main():
    args = parse_args()
    interactive = sys.stdin.isatty()

    if args.model is None:
        if not interactive:
            sys.exit("--model is required when not running interactively")
        print("Models:\n  1) cldrive  - CL-Drive cognitive load (ECG, EDA, EEG, Gaze)\n"
              "  2) drozy    - DROZY drowsiness (EEG, ECG, EOG, EMG, Face)")
        pick = ask("Which model", ['1', '2', 'cldrive', 'drozy'], default='1')
        args.model = {'1': 'cldrive', '2': 'drozy'}.get(pick, pick)
    if args.device is None:
        args.device = ask("Device (auto / cuda / cpu)", ['auto', 'cuda', 'cpu'], default='auto') \
            if interactive else 'auto'

    spec = DATASETS[args.model]
    wpath = args.weights or weights_path(args.model)
    if not os.path.exists(wpath):
        sys.exit(f"Weights not found: {wpath}\nTrain them first:  python train_final.py --dataset {args.model}")

    print(f"Loading {spec['title']} recordings...")
    arc = load_archive(args.model, locate_npz(args.model, args.data_path))
    subjects = sorted(int(s) for s in np.unique(arc['participant']))

    if args.subject is None:
        args.subject = ask(f"{spec['subject_name'].capitalize()} to replay ({', '.join(map(str, subjects))})",
                           subjects, default=subjects[0], cast=int) if interactive else subjects[0]
    elif args.subject not in subjects:
        sys.exit(f"Unknown {spec['subject_name']} {args.subject}. Available: {subjects}")
    if args.speed is None:
        args.speed = ask("Replay speed (1 = real time)", default=1.0, cast=float) if interactive else 1.0
    if args.speed <= 0 or args.interval <= 0:
        sys.exit("--speed and --interval must be positive")

    segments = replay_segments(arc, args.subject)
    modalities = spec['modalities']
    seg_lengths = [s['X'].shape[1] for s in segments]
    truth = [(s['levels'], s['y3']) for s in segments]
    total_s = sum(seg_lengths) / 100
    print(f"Replaying {spec['subject_name']} {args.subject}: {len(segments)} segment(s), "
          f"{total_s / 60:.1f} min of recording ({total_s / 60 / args.speed:.1f} min at {args.speed:g}x)")

    # 'spawn' keeps CUDA confined to the inference process and works the same on every OS
    ctx_mp = mp.get_context('spawn')
    sensor_q, result_q = ctx_mp.Queue(), ctx_mp.Queue()
    stop_event, go_event = ctx_mp.Event(), ctx_mp.Event()
    start_time = ctx_mp.Value('d', 0.0)

    from driver_state.sensors import sensor_process
    from driver_state.inference import inference_process

    infer = ctx_mp.Process(
        target=inference_process, name="inference", daemon=True,
        args=(wpath, args.device, modalities, seg_lengths, truth, args.interval,
              sensor_q, result_q, stop_event))
    sensors = [
        ctx_mp.Process(
            target=sensor_process, name=f"sensor-{m}", daemon=True,
            args=(m, [s['X'][a:b] for s in segments], CHUNK_SAMPLES, args.speed,
                  sensor_q, go_event, start_time, stop_event))
        for m, (a, b, _) in modalities.items()
    ]

    print(f"Starting inference process (device: {args.device}) and {len(sensors)} sensor processes...")
    infer.start()
    for p in sensors:
        p.start()

    # Wait for the model before the sensors start streaming
    while True:
        try:
            kind, payload = result_q.get(timeout=0.5)
        except queue.Empty:
            if not infer.is_alive():
                stop_event.set()
                sys.exit("Inference process exited before becoming ready.")
            continue
        if kind == 'error':
            stop_event.set()
            go_event.set()
            sys.exit(payload)
        if kind == 'ready':
            info = payload
            break

    run_ctx = {
        'title': spec['title'], 'class_names': spec['class_names'], 'level_name': spec['level_name'],
        'subject_name': spec['subject_name'], 'subject': args.subject, 'speed': args.speed,
        'interval': args.interval, 'modalities': modalities, 'modality_labels': spec['modality_labels'],
        'sessions': [s['session'] for s in segments], 'device': info['device'],
    }

    screen = Screen(args.plain)
    state = {'status': 'STARTING', 'device': info['device']}
    start_time.value = time.time()
    go_event.set()

    final = None
    try:
        while True:
            try:
                kind, payload = result_q.get(timeout=0.5)
            except queue.Empty:
                if not infer.is_alive():
                    break
                continue
            if kind == 'error':
                screen.close()
                print(payload)
                break
            state = payload
            screen.draw(render(state, run_ctx), state, run_ctx)
            if kind == 'done':
                final = state
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        go_event.set()
        for p in sensors + [infer]:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        screen.close()

    scored = state.get('scored', 0)
    print("\nReplay finished." if final else "\nStopped.")
    if scored:
        print(f"{scored} updates; smoothed class matched the ground truth in "
              f"{100 * state['agree'] / scored:.0f}% of them (subject seen in training).")


if __name__ == '__main__':
    main()
