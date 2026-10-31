"""
Finger-tapping analysis (modelled on MDS-UPDRS 3.4).

Input: continuous thumb-tip <-> index-tip aperture (px) plus hand size (px) per frame.

What changed vs v1 and why:
  * Aperture is divided by hand size (wrist -> middle MCP), so moving the hand
    closer to / further from the camera no longer changes the result.
  * The 11-frame Savitzky-Golay smoother (which removed ~90 % of a 4.5 Hz tap
    signal at 30 fps, so healthy fast tappers failed) is replaced by a zero-phase
    Butterworth low-pass whose cutoff sits well above the fastest human tapping.
  * Valleys are taken as the minimum between consecutive peaks, so every tap is
    a well-defined open-close cycle.
  * Decrement (the sequence effect, the hallmark PD sign) is measured as the
    change between the first and last third of the taps, plus a slope over time.
    v1 normalised a per-tap slope by the mean, which reported a 50 % decay as ~2 %.
  * Tap rate uses the real recording span from timestamps, not the client's claim.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from neurotrack.core import constants as K
from neurotrack.core.quality import AnalysisError, assess, require_ok
from neurotrack.core.signal import (
    choose_resample_rate,
    downsample_series,
    lowpass,
    resample_uniform,
    sanitize,
)

MAX_TAP_HZ = 8.0                 # fastest plausible voluntary tapping
SMOOTH_CUTOFF_HZ = 12.0          # low-pass cutoff, capped at 0.4 * fs
MIN_PROMINENCE_NORM = 0.08       # min open-close excursion, fraction of hand size
MIN_PROMINENCE_PX = 6.0          # fallback when hand size is not provided
HESITATION_FACTOR = 2.0          # interval > 2x median interval = hesitation
HEALTHY_RATE_HZ = 4.0


def _rel_change(x: np.ndarray) -> float:
    """(mean of last third - mean of first third) / mean of first third, in [-1, 1]."""
    n = len(x)
    if n < 6:
        return 0.0
    k = n // 3
    first, last = float(np.mean(x[:k])), float(np.mean(x[-k:]))
    if first <= 1e-9:
        return 0.0
    return float(np.clip((last - first) / first, -1.0, 1.0))


def _slope_pct_per_s(t: np.ndarray, x: np.ndarray) -> float:
    """Linear-trend slope as % of the fitted starting value per second."""
    if len(x) < 3 or np.ptp(t) <= 0:
        return 0.0
    slope, intercept = np.polyfit(t - t[0], x, 1)
    if abs(intercept) < 1e-9:
        return 0.0
    return float(slope / intercept * 100.0)


def analyze_finger_tap(frames: list[dict], *, expected_duration_s: float | None, reported_fps: float | None = None) -> dict:
    """frames: [{"v": aperture_px, "t": ms, "s": hand_size_px (optional)}]"""
    t_ms = [f["t"] for f in frames]
    v_px = [f["v"] for f in frames]
    s_px = [f.get("s") if f.get("s") is not None else np.nan for f in frames]
    have_size = np.isfinite(np.asarray(s_px, dtype=float)).sum() >= 0.5 * len(frames)

    san = sanitize(t_ms, v_px, s_px if have_size else np.zeros(len(frames)))
    q = assess(
        san,
        expected_duration_s=expected_duration_s,
        min_fps=K.MIN_TAPPING_FPS,
        min_coverage=K.MIN_COVERAGE,
        max_gap_s=K.MAX_GAP_S,
        min_samples=40,
        reported_fps=reported_fps,
    )
    require_ok(q, "finger-tap test")

    t_s, v = san.t_s, san.values[0]
    if have_size:
        sizes = san.values[1]
        valid = sizes[sizes > 1]
        hand_size = float(np.median(valid)) if len(valid) else float("nan")
        if not np.isfinite(hand_size) or hand_size <= 1:
            raise AnalysisError("Hand size could not be measured; keep the whole hand in view.")
        v = v / hand_size
        min_prom = MIN_PROMINENCE_NORM
        unit = "hand"
    else:
        hand_size = None
        min_prom = MIN_PROMINENCE_PX
        unit = "px"
        q.warnings.append("hand size not provided: amplitude is in pixels and depends on camera distance")

    fs = choose_resample_rate(t_s)
    tg, vg = resample_uniform(t_s, v, fs)
    vf = lowpass(vg, min(SMOOTH_CUTOFF_HZ, 0.4 * fs), fs)
    vel = np.gradient(vf) * fs   # aperture units per second

    robust_range = float(np.percentile(vf, 95) - np.percentile(vf, 5))
    prominence = max(0.25 * robust_range, min_prom)
    peaks, _ = find_peaks(vf, prominence=prominence, distance=max(1, int(fs / MAX_TAP_HZ)))
    if len(peaks) < 3:
        raise AnalysisError(
            "Could not detect taps. Keep the hand in view and tap thumb and index finger as wide and fast as you can.",
            [f"{len(peaks)} taps detected"],
        )

    # Valley before each peak (minimum between the previous peak and this one).
    starts = np.r_[0, peaks[:-1]]
    valleys_before = np.array([s + int(np.argmin(vf[s:p + 1])) for s, p in zip(starts, peaks)])
    ends = np.r_[peaks[1:], len(vf) - 1]
    valleys_after = np.array([p + int(np.argmin(vf[p:e + 1])) for p, e in zip(peaks, ends)])

    amps = vf[peaks] - vf[valleys_before]                      # opening amplitude per tap
    open_speed = np.array([vel[a:p + 1].max() if p > a else 0.0 for a, p in zip(valleys_before, peaks)])
    close_speed = np.array([-vel[p:b + 1].min() if b > p else 0.0 for p, b in zip(peaks, valleys_after)])
    speed = np.maximum(open_speed, close_speed)
    tap_t = tg[peaks]

    duration = float(tg[-1] - tg[0])
    n_taps = len(peaks)
    tap_rate = n_taps / duration if duration > 0 else 0.0

    itis = np.diff(tap_t)
    median_iti = float(np.median(itis))
    rhythm_cv = float(np.std(itis) / np.mean(itis)) if len(itis) > 1 else 0.0
    hes_mask = itis > HESITATION_FACTOR * median_iti
    hesitation_count = int(hes_mask.sum())
    arrest_ratio = float(np.sum(itis[hes_mask] - median_iti) / duration) if hesitation_count else 0.0

    amplitude_decay = _rel_change(amps)
    speed_decay = _rel_change(speed)
    amp_slope = _slope_pct_per_s(tap_t, amps)
    speed_slope = _slope_pct_per_s(tap_t, speed)
    mean_amp = float(np.mean(amps))
    decrement_index = float(np.clip(
        0.4 * max(0.0, -amplitude_decay) + 0.3 * max(0.0, -speed_decay) + 0.3 * min(1.0, rhythm_cv), 0.0, 1.0))

    # ── Component scores (0-100, 100 = healthiest). Uncalibrated, see constants.py.
    comp = {
        "rate":       float(np.clip((tap_rate - 1.0) / (HEALTHY_RATE_HZ - 1.0), 0, 1) * 100),
        "amp_decay":  float(np.clip(100 + amplitude_decay * 200, 0, 100)),
        "speed_decay": float(np.clip(100 + speed_decay * 200, 0, 100)),
        "rhythm":     float(np.clip(100 - (rhythm_cv - 0.10) * 250, 0, 100)),
        "hesitation": float(np.clip(100 - hesitation_count * 20 - arrest_ratio * 100, 0, 100)),
    }
    weights = {"rate": 0.30, "amp_decay": 0.20, "speed_decay": 0.15, "rhythm": 0.20, "hesitation": 0.15}
    if have_size:   # amplitude size (hypokinesia) is only meaningful when normalised
        comp["size"] = float(np.clip((mean_amp - 0.15) / 0.35, 0, 1) * 100)
        weights = {"rate": 0.25, "size": 0.15, "amp_decay": 0.20, "speed_decay": 0.15, "rhythm": 0.15, "hesitation": 0.10}
    score = K.blend_score(comp, weights)

    deficits = []
    if tap_rate < 2.5:
        deficits.append(("slowness", (HEALTHY_RATE_HZ - tap_rate) / HEALTHY_RATE_HZ))
    if amplitude_decay < -0.20:
        deficits.append(("amplitude decrement", -amplitude_decay))
    if speed_decay < -0.20:
        deficits.append(("speed decrement", -speed_decay))
    if rhythm_cv > 0.25:
        deficits.append(("irregular rhythm", rhythm_cv))
    if hesitation_count >= 2:
        deficits.append(("hesitations", min(1.0, 0.2 * hesitation_count + arrest_ratio)))
    if have_size and mean_amp < 0.3:
        deficits.append(("small amplitude", (0.3 - mean_amp) / 0.3))
    dominant = max(deficits, key=lambda d: d[1])[0] if deficits else "none"

    tap_events = [
        {"t": round(float((tap_t[i] - tg[0]) * 1000), 1), "amp": round(float(vf[peaks[i]]), 4),
         "opening": round(float(amps[i]), 4), "speed": round(float(speed[i]), 4)}
        for i in range(n_taps)
    ]

    return {
        "duration_s": round(duration, 2),
        "tap_count": n_taps,
        "overall_score": score,
        "severity": K.label_for(score),
        "dominant_feature": dominant,
        "amplitude_unit": unit,
        "hand_size_px": round(hand_size, 1) if hand_size else None,
        "biomarkers": {
            "tap_rate": round(tap_rate, 3),
            "mean_amplitude": round(mean_amp, 4),
            "mean_speed": round(float(np.mean(speed)), 4),
            "amplitude_decay": round(amplitude_decay, 4),
            "speed_decay": round(speed_decay, 4),
            "amplitude_slope_pct_per_s": round(amp_slope, 3),
            "speed_slope_pct_per_s": round(speed_slope, 3),
            "rhythm_cv": round(rhythm_cv, 4),
            "hesitation_count": hesitation_count,
            "arrest_ratio": round(arrest_ratio, 4),
            "decrement_index": round(decrement_index, 4),
        },
        "component_scores": {k: round(v, 1) for k, v in comp.items()},
        "tap_events": tap_events,
        "raw_frames": downsample_series(tg - tg[0], vf),
        "quality": q.to_dict(),
        "algorithm_version": K.ALGORITHM_VERSION,
    }
