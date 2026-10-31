"""Synthetic recordings with known ground truth, used by the regression tests."""
import math

import numpy as np


def frame_times(fps, dur_s, jitter=0.15, drop=0.0, rng=None):
    """Irregular frame timestamps (ms) like a real webcam + MediaPipe pipeline."""
    rng = rng or np.random.default_rng(0)
    t, out = 0.0, []
    while t < dur_s * 1000:
        if rng.random() >= drop:
            out.append(t)
        t += 1000.0 / fps * max(0.3, 1 + rng.normal(0, jitter))
    return out


def tapping(freq=4.5, fps=30, dur=10, hand_px=80.0, open_ratio=1.0, decay=0.0, noise_px=1.0,
            pauses=(), seed=0, with_size=True):
    """Thumb-index aperture: opens to open_ratio*hand size, linear amplitude decay."""
    rng = np.random.default_rng(seed)
    frames = []
    phase = 0.0
    last = 0.0
    for t in frame_times(fps, dur, rng=rng):
        dt = (t - last) / 1000.0
        last = t
        paused = any(a <= t / 1000 < b for a, b in pauses)
        if not paused:
            phase += 2 * math.pi * freq * dt
        amp = hand_px * open_ratio * (1 - decay * t / (dur * 1000))
        v = 0.08 * hand_px + amp * (0.5 - 0.5 * math.cos(phase)) + rng.normal(0, noise_px)
        f = {"v": max(0.0, v), "t": t}
        if with_size:
            f["s"] = hand_px + rng.normal(0, 0.5)
        frames.append(f)
    return frames


def hand(freq=5.0, amp_cm=1.0, fps=30, dur=15, hand_px=60.0, jitter_px=0.6, drop=0.0, seed=1,
         palm_cm=9.5):
    """Centroid oscillating along a 30 deg axis with peak-to-peak amp_cm, plus drift + jitter."""
    rng = np.random.default_rng(seed)
    amp_px = amp_cm / palm_cm * hand_px / 2.0       # half of peak-to-peak
    out = []
    for t in frame_times(fps, dur, drop=drop, rng=rng):
        s = t / 1000.0
        osc = amp_px * math.sin(2 * math.pi * freq * s)
        drift = 3.0 * math.sin(2 * math.pi * 0.2 * s)
        out.append({
            "x": 160 + osc * math.cos(0.5) + drift + rng.normal(0, jitter_px),
            "y": 120 + osc * math.sin(0.5) + rng.normal(0, jitter_px),
            "s": hand_px + rng.normal(0, 0.5),
            "t": t,
        })
    return out


def spiral(turns=3.5, max_r=140.0, start_angle=0.0, scale=1.0, dur=8.0, hz=60, cx=320, cy=160,
           tremor_hz=0.0, tremor_px=0.0, micro=0.0, dup_every=0, ms_resolution=True, seed=2):
    """Archimedean spiral; `micro` < 0 makes later loops progressively tighter."""
    rng = np.random.default_rng(seed)
    n = int(dur * hz)
    pts = []
    total = turns * 2 * math.pi
    for i in range(n):
        u = i / (n - 1)
        a = u * total
        # spacing changes linearly from 1 to (1+micro) along the drawing
        r = max_r * scale * (u + 0.5 * micro * u * u) / (1 + 0.5 * micro)
        t = 1000 + u * dur * 1000
        r += tremor_px * math.sin(2 * math.pi * tremor_hz * (t / 1000.0))
        ang = start_angle + a
        t = math.floor(t) if ms_resolution else t
        x, y = cx + r * math.cos(ang) + rng.normal(0, 0.3), cy + r * math.sin(ang) + rng.normal(0, 0.3)
        pts.append({"x": x, "y": y, "t": t})
        if dup_every and i % dup_every == 0:
            pts.append({"x": x + 0.5, "y": y, "t": t})
    return pts
