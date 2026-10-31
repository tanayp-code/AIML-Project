#!/usr/bin/env python3
"""
Compare resource use of the PyTorch (.pt) and ONNX Runtime (.onnx) versions of the final models.

  ./benchmark_runtime.sh            # installs onnx + onnxruntime if needed, then runs this
  python benchmark_runtime.py --runs 500

Steps:
  1. Exports weights/<model>_final.pt to weights/<model>_final.onnx (checks outputs match)
  2. Runs every runtime/device combination in a fresh subprocess, so memory numbers are not
     mixed: import time, load time, RAM, GPU memory, latency per 10-second window
  3. Prints a table, plus install size and model file size

Needs `onnx` (export) and `onnxruntime` (inference). They are looked up after the normal
site-packages, so a copy installed by benchmark_runtime.sh never shadows your venv packages.
"""

import os
import sys
import json
import math
import time
import argparse
import subprocess

LIBS = os.environ.get('ONNX_BENCH_LIBS')
if LIBS and os.path.isdir(LIBS):
    sys.path.append(LIBS)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ['cldrive', 'drozy']


def rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return float('nan')


def peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ---------------------------------------------------------------------------
# Worker: one runtime, one device, one model, in a clean process
# ---------------------------------------------------------------------------

def worker(runtime, device, model_key, num_channels, runs):
    import numpy as np
    out = {'runtime': runtime, 'device': device, 'model': model_key, 'rss_start_mb': rss_mb()}
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, num_channels, 1000)).astype(np.float32)

    t0 = time.perf_counter()
    if runtime == 'pytorch':
        import torch
        out['import_s'] = time.perf_counter() - t0
        out['rss_after_import_mb'] = rss_mb()
        sys.path.insert(0, HERE)
        from driver_state.models import load_checkpoint
        dev = torch.device(device)
        t1 = time.perf_counter()
        model, _ = load_checkpoint(os.path.join(HERE, 'weights', f'{model_key}_final.pt'), dev)
        xt = torch.from_numpy(x).to(dev)

        def run():
            with torch.no_grad():
                logits, level = model(xt)
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
            return logits.cpu().numpy(), level.cpu().numpy()
    else:
        import onnxruntime as ort
        out['import_s'] = time.perf_counter() - t0
        out['rss_after_import_mb'] = rss_mb()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' \
            else ['CPUExecutionProvider']
        t1 = time.perf_counter()
        sess = ort.InferenceSession(os.path.join(HERE, 'weights', f'{model_key}_final.onnx'),
                                    providers=providers)
        out['providers_used'] = sess.get_providers()

        def run():
            logits, level = sess.run(None, {'x': x})
            return logits, level

    first = time.perf_counter()
    logits, level = run()
    out['load_s'] = first - t1
    out['first_run_ms'] = (time.perf_counter() - first) * 1000

    for _ in range(10):  # warm-up
        run()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        run()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    out['latency_mean_ms'] = sum(times) / len(times)
    out['latency_p95_ms'] = times[int(0.95 * (len(times) - 1))]
    out['rss_end_mb'] = rss_mb()
    out['peak_rss_mb'] = peak_rss_mb()
    out['logits'] = [float(v) for v in logits.ravel()]
    out['level'] = float(np.asarray(level).ravel()[0])

    if runtime == 'pytorch' and device == 'cuda':
        out['gpu_mem_mb'] = torch.cuda.max_memory_allocated() / 2**20
    print('RESULT ' + json.dumps(out))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(model_key):
    import torch
    import torch.nn as nn
    sys.path.insert(0, HERE)
    from driver_state.models import load_checkpoint

    class MatrixPool(nn.Module):
        """Exact replacement for AdaptiveAvgPool1d at a fixed input length (ONNX cannot
        export adaptive pooling when the output size does not divide the input size)."""
        def __init__(self, length, out_size):
            super().__init__()
            P = torch.zeros(length, out_size)
            for i in range(out_size):
                s, e = (i * length) // out_size, math.ceil((i + 1) * length / out_size)
                P[s:e, i] = 1.0 / (e - s)
            self.register_buffer('P', P)

        def forward(self, x):
            return x @ self.P

    pt_path = os.path.join(HERE, 'weights', f'{model_key}_final.pt')
    onnx_path = os.path.join(HERE, 'weights', f'{model_key}_final.onnx')
    model, ckpt = load_checkpoint(pt_path, torch.device('cpu'))
    C = ckpt['num_channels']

    x = torch.randn(4, C, 1000)
    with torch.no_grad():
        ref = model(x)
    # Length reaching the pooling layer for a 1000-sample window: 1000 -> 500 -> 250 -> 125 -> 62
    for stem in model.stems.values():
        stem.adaptive_pool = MatrixPool(62, stem.adaptive_pool.output_size)
    with torch.no_grad():
        new = model(x)
    diff = max((a - b).abs().max().item() for a, b in zip(ref, new))
    if diff > 1e-4:
        raise RuntimeError(f"Pooling replacement changed outputs by {diff}")

    torch.onnx.export(model, (x[:1],), onnx_path, input_names=['x'], output_names=['logits', 'level'],
                      dynamic_axes={'x': {0: 'batch'}, 'logits': {0: 'batch'}, 'level': {0: 'batch'}},
                      opset_version=17)
    return onnx_path, C


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 2**20


def package_dir(name):
    import importlib.util
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def main():
    p = argparse.ArgumentParser(description="PyTorch vs ONNX Runtime resource comparison")
    p.add_argument('--runs', type=int, default=300, help="Timed inferences per configuration")
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--worker', nargs=4, metavar=('RUNTIME', 'DEVICE', 'MODEL', 'CHANNELS'),
                   help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        rt, dev, mk, ch = args.worker
        worker(rt, dev, mk, int(ch), args.runs)
        return

    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError as exc:
        sys.exit(f"Missing package ({exc}). Run ./benchmark_runtime.sh, or: pip install onnx onnxruntime")

    import warnings
    warnings.filterwarnings('ignore')
    sys.path.insert(0, HERE)
    from driver_state.device import resolve_device

    print("Exporting ONNX models...")
    channels = {}
    for mk in args.models:
        path, channels[mk] = export_onnx(mk)
        print(f"  {path}")

    configs = [('pytorch', 'cpu'), ('onnxruntime', 'cpu')]
    gpu, note = resolve_device('cuda')
    if gpu.type == 'cuda':
        configs.insert(1, ('pytorch', 'cuda'))
    else:
        print(f"  PyTorch GPU skipped: {note}")
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        configs.append(('onnxruntime', 'cuda'))
    else:
        print("  ONNX Runtime GPU skipped: CPU-only 'onnxruntime' installed "
              "(GPU needs the 'onnxruntime-gpu' package)")

    results = []
    print(f"\nBenchmarking ({args.runs} timed runs each, batch of 1 window, each in a fresh process)...")
    for mk in args.models:
        for rt, dev in configs:
            print(f"  {mk:<8} {rt:<12} {dev}", end='', flush=True)
            cmd = [sys.executable, os.path.abspath(__file__), '--runs', str(args.runs),
                   '--worker', rt, dev, mk, str(channels[mk])]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            line = next((l for l in proc.stdout.splitlines() if l.startswith('RESULT ')), None)
            if line is None:
                print("  FAILED\n" + proc.stderr[-800:])
                continue
            r = json.loads(line[7:])
            results.append(r)
            print(f"  done ({r['latency_mean_ms']:.2f} ms)")

    # Table
    print("\n" + "=" * 104)
    print(f"{'Model':<9}{'Runtime':<13}{'Device':<7}{'Import s':>9}{'Load s':>8}{'First ms':>10}"
          f"{'Mean ms':>9}{'p95 ms':>8}{'RAM MB':>9}{'Peak MB':>9}{'GPU MB':>8}{'Max diff':>10}")
    print("-" * 104)
    for mk in args.models:
        ref = next((r for r in results if r['model'] == mk and r['runtime'] == 'pytorch'
                    and r['device'] == 'cpu'), None)
        for r in (r for r in results if r['model'] == mk):
            diff = max(abs(a - b) for a, b in zip(r['logits'], ref['logits'])) if ref else float('nan')
            gpu_mb = f"{r['gpu_mem_mb']:.0f}" if 'gpu_mem_mb' in r else "-"
            dev = r['device']
            # ONNX Runtime silently drops to CPU if its CUDA provider cannot start
            if r['runtime'] == 'onnxruntime' and r['device'] == 'cuda' \
                    and r.get('providers_used', [''])[0] != 'CUDAExecutionProvider':
                dev = 'cpu*'
            print(f"{r['model']:<9}{r['runtime']:<13}{dev:<7}{r['import_s']:>9.2f}{r['load_s']:>8.2f}"
                  f"{r['first_run_ms']:>10.1f}{r['latency_mean_ms']:>9.2f}{r['latency_p95_ms']:>8.2f}"
                  f"{r['rss_end_mb']:>9.0f}{r['peak_rss_mb']:>9.0f}{gpu_mb:>8}{diff:>10.1e}")
    print("=" * 104)
    print("RAM MB = resident memory after the timed runs; Peak MB = highest resident memory of the process;")
    print("GPU MB = memory PyTorch allocated on the GPU (the CUDA context adds roughly 200-400 MB more);")
    print("Max diff = largest logit difference from PyTorch CPU on the same input.")

    # Install and file sizes
    print("\nInstall size on disk")
    torch_dir = package_dir('torch')
    if torch_dir:
        site = os.path.dirname(torch_dir)
        extra = [os.path.join(site, d) for d in ('nvidia', 'triton') if os.path.isdir(os.path.join(site, d))]
        torch_total = dir_size_mb(torch_dir) + sum(dir_size_mb(d) for d in extra)
        print(f"  PyTorch (torch + bundled CUDA libraries): {torch_total:8.0f} MB")
    ort_dir = package_dir('onnxruntime')
    if ort_dir:
        print(f"  ONNX Runtime ({'GPU' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'CPU'} build):"
              f"               {dir_size_mb(ort_dir):8.0f} MB")
    print("\nModel file size")
    for mk in args.models:
        pt = os.path.getsize(os.path.join(HERE, 'weights', f'{mk}_final.pt')) / 2**20
        ox = os.path.getsize(os.path.join(HERE, 'weights', f'{mk}_final.onnx')) / 2**20
        print(f"  {mk:<8} .pt {pt:5.2f} MB   .onnx {ox:5.2f} MB")


if __name__ == '__main__':
    main()
