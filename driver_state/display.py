"""
Terminal dashboard (plain text, redrawn in place with ANSI cursor codes).
"""

import sys
import time

BAR_WIDTH = 24


def _bar(p):
    n = int(round(p * BAR_WIDTH))
    return "#" * n + "-" * (BAR_WIDTH - n)


def _clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def render(state, ctx):
    """state: last message from the inference process; ctx: static run information."""
    names = ctx['class_names']
    lines = [
        f"DRIVER STATE MONITOR  |  {ctx['title']} model",
        f"Device: {state.get('device', ctx.get('device', '?'))}",
        f"Replay: {ctx['subject_name']} {ctx['subject']} (part of the training data)   "
        f"Speed: {ctx['speed']:g}x   Update every {ctx['interval']:g} s",
    ]
    seg_info = ""
    if state.get('num_segments', 1) > 1:
        seg_info = f"   Session {ctx['sessions'][state['segment']]} ({state['segment'] + 1} of {state['num_segments']})"
    lat = state.get('latency_ms')
    lines.append(f"Stream time: {_clock(state.get('stream_seconds', 0))}{seg_info}   "
                 f"Status: {state.get('status', 'STARTING')}"
                 + (f"   Inference: {lat:.1f} ms" if lat is not None else ""))
    lines.append("")

    lines.append(f"{'SENSORS (simulated)':<24}{'channels':>9}{'samples':>11}   last packet")
    now = time.time()
    for m, (a, b, _) in ctx['modalities'].items():
        s = state.get('sensors', {}).get(m, {})
        if s.get('done'):
            age = "finished"
        elif s.get('last_rx') is None:
            age = "waiting"
        else:
            age = f"{now - s['last_rx']:.2f} s ago"
        lines.append(f"  {ctx['modality_labels'][m]:<22}{b - a:>9}{s.get('samples', 0):>11,}   {age}")
    lines.append("")

    pred = state.get('prediction')
    lines.append("CURRENT ESTIMATE (latest 10 s window)")
    if pred is None:
        lines += [f"  {n:<9} |{'-' * BAR_WIDTH}|    -" for n in names]
        lines += ["  Class   : -", f"  Level   : -", "  Truth   : -"]
    else:
        for i, n in enumerate(names):
            lines.append(f"  {n:<9} |{_bar(pred['probs'][i])}| {100 * pred['probs'][i]:3.0f}%")
        lines.append(f"  Class   : {names[pred['cls']]}   (smoothed over ~20 s: {names[pred['smoothed_cls']]})")
        lines.append(f"  Level   : {pred['level']:.1f} / 9 {ctx['level_name']}")
        lines.append(f"  Truth   : {ctx['level_name']} {pred['true_level']} ({names[pred['true_cls']]})")
    lines.append("")

    hist = state.get('history', [])
    letters = "".join(names[c][0] for c in hist)
    key = ", ".join(f"{n[0]} = {n}" for n in names)
    lines.append(f"HISTORY (smoothed class, newest on the right; {key})")
    lines.append(f"  {letters if letters else '-'}")
    scored = state.get('scored', 0)
    if scored:
        lines.append(f"  Matches truth: {100 * state['agree'] / scored:.0f}% of {scored} updates")
    else:
        lines.append("  Matches truth: -")
    lines.append("")
    lines.append("Press Ctrl+C to stop")
    return lines


class Screen:
    def __init__(self, plain: bool):
        self.plain = plain or not sys.stdout.isatty()
        self.started = False

    def draw(self, lines, state=None, ctx=None):
        if self.plain:
            pred = (state or {}).get('prediction')
            names = ctx['class_names'] if ctx else []
            msg = f"[{_clock((state or {}).get('stream_seconds', 0))}] {(state or {}).get('status', '')}"
            if pred:
                msg += (f" | {names[pred['cls']]:<8} p={max(pred['probs']):.2f}"
                        f" smoothed={names[pred['smoothed_cls']]:<8} level={pred['level']:.1f}"
                        f" | truth {pred['true_level']} ({names[pred['true_cls']]})")
            print(msg, flush=True)
            return
        if not self.started:
            sys.stdout.write("\x1b[?25l\x1b[2J")
            self.started = True
        sys.stdout.write("\x1b[H" + "\n".join(line + "\x1b[K" for line in lines) + "\n\x1b[J")
        sys.stdout.flush()

    def close(self):
        if not self.plain and self.started:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()
