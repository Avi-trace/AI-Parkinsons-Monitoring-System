"""
Shared signal-processing primitives.

Every analyzer goes through the same path:
    raw (t, value...) samples  ->  sanitize  ->  uniform resample  ->  filter / Welch PSD

Webcam frames and pointer events are NOT evenly spaced (dropped frames, variable
MediaPipe latency, coalesced input events). An FFT on unevenly spaced samples
with an "average" sample rate smears and shifts spectral peaks, so all spectral
work here happens on a uniform grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import butter, sosfiltfilt, welch

from neurotrack.core.constants import NYQUIST_MARGIN


@dataclass
class Sanitized:
    t_s: np.ndarray            # seconds, strictly increasing
    values: list[np.ndarray]   # one array per input channel, same length as t_s
    n_dropped: int             # non-finite samples removed
    n_merged: int              # duplicate-timestamp samples merged
    reordered: bool            # input timestamps were not monotonic


def sanitize(t_ms, *channels) -> Sanitized:
    """Drop non-finite samples, sort by time, merge samples sharing a timestamp."""
    t = np.asarray(t_ms, dtype=float)
    chans = [np.asarray(c, dtype=float) for c in channels]
    finite = np.isfinite(t)
    for c in chans:
        finite &= np.isfinite(c)
    n_dropped = int((~finite).sum())
    t = t[finite]
    chans = [c[finite] for c in chans]

    reordered = bool(len(t) > 1 and np.any(np.diff(t) < 0))
    order = np.argsort(t, kind="stable")
    t = t[order]
    chans = [c[order] for c in chans]

    # Merge identical timestamps (e.g. two pointer events in the same ms) by
    # averaging, instead of creating a ~0 dt that explodes velocity estimates.
    uniq, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    n_merged = int(len(t) - len(uniq))
    if n_merged:
        chans = [np.bincount(inverse, weights=c) / counts for c in chans]
    return Sanitized(uniq / 1000.0, chans, n_dropped, n_merged, reordered)


def effective_rate(t_s: np.ndarray) -> float:
    """Robust sample rate: 1 / median inter-sample interval."""
    if len(t_s) < 2:
        return 0.0
    dt = np.diff(t_s)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else 0.0


def resample_uniform(t_s: np.ndarray, values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Cubic-spline interpolation onto an evenly spaced grid at `fs` Hz.

    Linear interpolation behaves like a sinc^2 low-pass: at 30 fps it removes ~9 %
    of a 5 Hz tremor's amplitude (~35 % at 12 fps). A cubic spline is nearly flat
    up to ~0.4 fs. Falls back to linear for very short inputs."""
    if len(t_s) < 2 or fs <= 0:
        return t_s.copy(), values.copy()
    n = int(np.floor((t_s[-1] - t_s[0]) * fs)) + 1
    grid = t_s[0] + np.arange(n) / fs
    if len(t_s) < 4:
        return grid, np.interp(grid, t_s, values)
    return grid, make_interp_spline(t_s, values, k=3)(grid)


def choose_resample_rate(t_s: np.ndarray, cap: float = 120.0) -> float:
    """Resample at the measured rate (rounded), never inventing bandwidth we don't have."""
    return float(min(cap, max(1.0, round(effective_rate(t_s)))))


def max_resolvable_hz(fs: float) -> float:
    return fs / NYQUIST_MARGIN


def _sos(kind: str, cutoff, fs: float, order: int):
    nyq = fs / 2.0
    wn = np.atleast_1d(np.asarray(cutoff, dtype=float)) / nyq
    wn = np.clip(wn, 1e-4, 0.999)
    return butter(order, wn if len(wn) > 1 else wn[0], btype=kind, output="sos")


def _filtfilt(sos, x: np.ndarray) -> np.ndarray:
    padlen = min(3 * (2 * len(sos) + 1), len(x) - 1)
    if padlen < 1:
        return x.copy()
    return sosfiltfilt(sos, x, padlen=padlen)


def lowpass(x: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    if cutoff >= fs / 2.0 * 0.999 or len(x) < 8:
        return x.copy()
    return _filtfilt(_sos("lowpass", cutoff, fs, order), x)


def highpass(x: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    if len(x) < 8:
        return x - np.mean(x)
    return _filtfilt(_sos("highpass", cutoff, fs, order), x)


def bandpass(x: np.ndarray, lo: float, hi: float, fs: float, order: int = 3) -> np.ndarray:
    hi = min(hi, fs / 2.0 * 0.95)
    if hi <= lo or len(x) < 8:
        return np.zeros_like(x)
    return _filtfilt(_sos("bandpass", (lo, hi), fs, order), x)


def welch_psd(x: np.ndarray, fs: float, seg_seconds: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD with ~seg_seconds windows (0.25 Hz resolution at 4 s), 50 % overlap."""
    n = len(x)
    if n < 8:
        return np.array([0.0]), np.array([0.0])
    nperseg = int(min(n, max(16, round(seg_seconds * fs))))
    return welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear", window="hann")


def band_power(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    mask = (f >= lo) & (f <= hi)
    if mask.sum() < 2:
        return float(np.sum(p[mask]) * (f[1] - f[0] if len(f) > 1 else 0.0))
    return float(np.trapezoid(p[mask], f[mask]))


@dataclass
class Peak:
    freq_hz: float
    prominence: float   # peak PSD / median PSD inside the searched band


def spectral_peak(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> Peak | None:
    """Highest PSD bin inside [lo, hi] with parabolic interpolation of its frequency."""
    mask = (f >= lo) & (f <= hi)
    idx = np.flatnonzero(mask)
    if len(idx) < 3:
        return None
    k = idx[np.argmax(p[idx])]
    freq = f[k]
    if 0 < k < len(p) - 1:
        a, b, c = p[k - 1], p[k], p[k + 1]
        denom = a - 2 * b + c
        if denom != 0:
            freq = f[k] + 0.5 * (a - c) / denom * (f[1] - f[0])
    median = float(np.median(p[idx])) + 1e-20
    return Peak(float(freq), float(p[k] / median))


def principal_axis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Project 2-D motion onto its dominant direction (PCA). Tremor oscillates along
    one axis; analysing x and y separately splits its power in two."""
    pts = np.column_stack([x - x.mean(), y - y.mean()])
    if len(pts) < 3:
        return pts[:, 0]
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    return pts @ vt[0]


def amplitude_spectrum(f: np.ndarray, p: np.ndarray, fmax: float, max_bins: int = 80) -> tuple[list[float], float]:
    """sqrt(PSD) up to fmax for frontend charts (list, Hz per bin)."""
    keep = f <= fmax
    spec = np.sqrt(np.maximum(p[keep], 0.0))[:max_bins]
    res = float(f[1] - f[0]) if len(f) > 1 else 0.0
    return [round(float(v), 6) for v in spec], round(res, 4)


def downsample_series(t_s: np.ndarray, v: np.ndarray, max_points: int = 300) -> list[dict]:
    step = max(1, int(np.ceil(len(v) / max_points)))
    return [{"t": round(float(t * 1000.0), 1), "v": round(float(val), 4)} for t, val in zip(t_s[::step], v[::step])]
