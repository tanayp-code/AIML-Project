"""
Sensor simulator process (numpy only, never imports torch).

Each modality runs in its own process and streams its channels in small timestamped
packets at the real sampling rate (optionally sped up), the way a Bluetooth/USB device
would. The samples are replayed from the preprocessed recordings at 100 Hz.

Packet format sent on the shared queue:
    ('chunk', modality, segment_index, first_sample_index, data[C, n])
    ('eof', modality)
"""

import time
import signal

from .config import SAMPLE_RATE


def sensor_process(modality, segments, chunk_samples, speed, out_q, go_event, start_time, stop_event):
    """
    segments: list of (C, T) float32 arrays for this modality, one per recording segment.
    start_time: shared double set by the launcher just before go_event, so every sensor
    shares the same stream clock.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the launcher handles Ctrl+C
    go_event.wait()
    t0 = start_time.value
    sent = 0  # samples sent so far across all segments (stream clock)

    for seg_idx, data in enumerate(segments):
        T = data.shape[1]
        for s in range(0, T, chunk_samples):
            n = min(chunk_samples, T - s)
            due = t0 + (sent + n) / SAMPLE_RATE / speed  # packet leaves once its last sample "happened"
            while True:
                if stop_event.is_set():
                    return
                wait = due - time.time()
                if wait <= 0:
                    break
                time.sleep(min(wait, 0.05))
            out_q.put(('chunk', modality, seg_idx, s, data[:, s:s + n].copy()))
            sent += n

    out_q.put(('eof', modality))
