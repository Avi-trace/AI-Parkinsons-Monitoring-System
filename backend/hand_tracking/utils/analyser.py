"""
Finger tap analyzer — continuous aperture signal approach.

Instead of a fragile state machine, we:
1. Record the raw thumb-index aperture signal continuously at ~30fps
2. Find peaks (open) and valleys (closed) via scipy.signal.find_peaks
3. Derive tap events from consecutive peak→valley→peak cycles
4. Compute all biomarkers from the resulting tap sequence

Biomarkers:
  - tap_rate          taps per second
  - mean_speed        mean closing speed (px/s) — how fast finger closes
  - amplitude_decay   linear slope of peak amplitudes over time
  - speed_decay       linear slope of closing speed over time
  - rhythm_cv         coefficient of variation of inter-tap intervals
  - hesitation_count  gaps > 2.5× mean ITI
  - arrest_ratio      fraction of time spent in hesitations
  - decrement_index   composite impairment (0–1, higher = worse)
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import List, Tuple
from hand_tracking.models.schema import FingerTapBiomarkers, FingerTapResult


def analyze_aperture(
    aperture_signal: List[dict],   # [{v: float, t: float}, ...] raw aperture stream
    duration_s: float,
) -> Tuple[FingerTapBiomarkers, int, str, str, List[dict]]:
    """
    aperture_signal: list of {v: aperture_px, t: timestamp_ms}
    Returns: biomarkers, score, severity, dominant_feature, tap_events
    """
    if len(aperture_signal) < 20:
        raise ValueError("Not enough frames for analysis")

    ts  = np.array([f["t"] for f in aperture_signal], dtype=float)
    vs  = np.array([f["v"] for f in aperture_signal], dtype=float)

    # ── 1. Smooth with Savitzky-Golay to reduce MediaPipe jitter ─────────
    win = min(11, len(vs) // 4 * 2 + 1)   # must be odd
    win = win if win % 2 == 1 else win - 1
    win = max(win, 5)
    vs_smooth = savgol_filter(vs, window_length=win, polyorder=2)

    # ── 2. Find peaks (open phases) and valleys (closed phases) ──────────
    # Dynamic prominence: 20% of signal range
    sig_range  = vs_smooth.max() - vs_smooth.min()
    prominence = max(sig_range * 0.20, 8.0)
    min_dist   = max(3, int(len(vs) / (duration_s * 8)))   # at most 8 Hz taps

    peaks,  _  = find_peaks( vs_smooth, prominence=prominence, distance=min_dist)
    valleys, _ = find_peaks(-vs_smooth, prominence=prominence * 0.5, distance=min_dist)

    if len(peaks) < 2:
        raise ValueError("Could not detect taps. Make sure your hand is clearly visible and tap widely.")

    # ── 3. Build tap list from peaks ──────────────────────────────────────
    # Each tap = one open-phase peak
    tap_times = ts[peaks]            # ms timestamps
    tap_amps  = vs_smooth[peaks]     # aperture at peak (px)

    # Closing speed for each tap: (peak_amp) / (time from peak to next valley)
    speeds = []
    for i, pk in enumerate(peaks):
        # Find the next valley after this peak
        next_valleys = valleys[valleys > pk]
        if len(next_valleys):
            nv = next_valleys[0]
            dt = (ts[nv] - ts[pk]) / 1000.0
            if dt > 0:
                speeds.append((tap_amps[i] - vs_smooth[nv]) / dt)
            else:
                speeds.append(np.nan)
        else:
            speeds.append(np.nan)
    speeds = np.array(speeds, dtype=float)
    valid_speeds = speeds[~np.isnan(speeds)]

    # ── 4. Biomarkers ─────────────────────────────────────────────────────
    n_taps   = len(tap_times)
    tap_rate = n_taps / duration_s

    mean_amp   = float(np.mean(tap_amps))
    mean_speed = float(np.mean(valid_speeds)) if len(valid_speeds) else 0.0

    # Amplitude decay: linear regression slope, normalised by mean amp
    x = np.arange(n_taps, dtype=float)
    amp_slope = float(np.polyfit(x, tap_amps, 1)[0])
    # Normalise: slope per tap / mean amp  → -1..0 range (negative = decaying)
    amplitude_decay = float(np.clip(amp_slope / (mean_amp + 1e-6), -1.0, 0.5))

    # Speed decay: same approach on closing speeds
    if len(valid_speeds) >= 2:
        sp_slope  = float(np.polyfit(np.arange(len(valid_speeds), dtype=float), valid_speeds, 1)[0])
        speed_decay = float(np.clip(sp_slope / (mean_speed + 1e-6), -1.0, 0.5))
    else:
        speed_decay = 0.0

    # Inter-tap intervals
    itis = np.diff(tap_times) / 1000.0   # seconds
    mean_iti = float(np.mean(itis)) if len(itis) else 0.0
    rhythm_cv = float(np.std(itis) / (mean_iti + 1e-6)) if len(itis) else 0.0

    # Hesitations: ITI > 2.5× mean
    hesitation_mask  = itis > 2.5 * mean_iti
    hesitation_count = int(np.sum(hesitation_mask))
    arrest_ratio     = float(np.sum(itis[hesitation_mask]) / duration_s) if hesitation_count else 0.0

    # Decrement index: composite of amplitude decay + speed decay + arrhythmia
    decrement_index = float(np.clip(
        (-amplitude_decay * 0.4) + (-speed_decay * 0.3) + (rhythm_cv * 0.3),
        0.0, 1.0
    ))

    # ── 5. Scoring ────────────────────────────────────────────────────────
    NORM_RATE = 4.5   # healthy: ~4–5 taps/s

    rate_score   = np.clip((tap_rate / NORM_RATE) * 100, 0, 100)
    amp_score    = np.clip(100 + amplitude_decay * 150, 0, 100)
    speed_score  = np.clip(100 + speed_decay * 150, 0, 100)
    rhythm_score = np.clip(100 - rhythm_cv * 120, 0, 100)
    hesit_score  = np.clip(100 - hesitation_count * 15 - arrest_ratio * 80, 0, 100)

    overall = (
        rate_score   * 0.30 +
        amp_score    * 0.20 +
        speed_score  * 0.20 +
        rhythm_score * 0.20 +
        hesit_score  * 0.10
    )
    overall_score = int(round(np.clip(overall, 0, 100)))

    # ── 6. Severity + dominant feature ────────────────────────────────────
    if overall_score >= 80:   severity = "Normal"
    elif overall_score >= 60: severity = "Mild"
    elif overall_score >= 40: severity = "Moderate"
    else:                     severity = "Severe"

    deficits = []
    if tap_rate      < 3.0:   deficits.append(("bradykinesia",  (NORM_RATE - tap_rate) / NORM_RATE))
    if amplitude_decay < -0.20: deficits.append(("amplitude decay", -amplitude_decay))
    if speed_decay   < -0.20: deficits.append(("speed decay",    -speed_decay))
    if rhythm_cv     > 0.25:  deficits.append(("arrhythmia",     rhythm_cv))
    if hesitation_count >= 2: deficits.append(("freezing",       arrest_ratio))
    dominant = max(deficits, key=lambda x: x[1])[0] if deficits else "none"

    biomarkers = FingerTapBiomarkers(
        tap_rate          = round(tap_rate, 2),
        mean_amplitude    = round(mean_amp, 1),
        mean_speed        = round(mean_speed, 1),
        amplitude_decay   = round(amplitude_decay, 4),
        speed_decay       = round(speed_decay, 4),
        rhythm_cv         = round(rhythm_cv, 4),
        hesitation_count  = hesitation_count,
        arrest_ratio      = round(arrest_ratio, 4),
        decrement_index   = round(decrement_index, 4),
    )

    # Tap events for frontend chart: {t, amp, speed}
    tap_events = [
        {"t": float(tap_times[i]), "amp": float(tap_amps[i]),
         "speed": float(speeds[i]) if not np.isnan(speeds[i]) else 0.0}
        for i in range(n_taps)
    ]

    return biomarkers, overall_score, severity, dominant, tap_events