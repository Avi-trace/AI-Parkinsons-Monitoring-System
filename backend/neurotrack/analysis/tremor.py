"""
Rest and postural tremor analysis (modelled on MDS-UPDRS 3.15 postural / 3.17 rest).

Input per hand: composite hand centroid (x, y px) + hand size (px) per frame.

What changed vs v1 and why:
  * Samples are resampled onto a uniform grid before any spectrum (dropped frames
    made v1's FFT unreliable).
  * Welch PSD on the principal motion axis instead of one FFT on x and y.
  * The tremor search band is capped at fs / 2.5. At low fps a real 8 Hz tremor
    aliased to "3 Hz" in v1; now frequency is only reported when the measured
    sample rate can actually resolve it.
  * A tremor needs BOTH a clear spectral peak (>= 5x the band median) AND an
    amplitude above landmark jitter. v1 reported pure 1 px jitter as a 4.4 Hz tremor.
  * Amplitude is reported in approximate cm and binned like MDS-UPDRS
    (<1, 1-3, 3-10, >=10 cm) instead of an arbitrary 0-100 formula.
  * A hand with no / poor tracking is "not analysed" — never "Normal". If no hand
    is usable the whole request is rejected.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from neurotrack.core import constants as K
from neurotrack.core.quality import AnalysisError, assess
from neurotrack.core.signal import (
    amplitude_spectrum,
    band_power,
    bandpass,
    choose_resample_rate,
    highpass,
    lowpass,
    max_resolvable_hz,
    principal_axis,
    resample_uniform,
    sanitize,
    spectral_peak,
    welch_psd,
)

GRADE_LABELS = ("Normal", "Slight", "Mild", "Moderate", "Severe")   # MDS-UPDRS 0-4 wording


def _continuous_grade(amp_cm: float) -> float:
    """Piecewise-linear 0-4 grade aligned to the MDS-UPDRS amplitude bins."""
    b1, b2, b3 = K.UPDRS_AMPLITUDE_BINS_CM
    if amp_cm <= 0:
        return 0.0
    if amp_cm < b1:
        return amp_cm / b1                       # 0-1
    if amp_cm < b2:
        return 1 + (amp_cm - b1) / (b2 - b1)     # 1-2
    if amp_cm < b3:
        return 2 + (amp_cm - b2) / (b3 - b2)     # 2-3
    return min(4.0, 3 + (amp_cm - b3) / b3)      # 3-4


def _onset_latency(band_sig: np.ndarray, fs: float, threshold_norm: float) -> float | None:
    """Seconds until the tremor envelope first stays above threshold for 1 s
    (re-emergent tremor appears after a latency in postural holding)."""
    env = np.abs(hilbert(band_sig)) * 2.0          # envelope -> peak-to-peak
    env = lowpass(env, 1.0, fs, order=2)
    above = env >= threshold_norm
    need = int(fs)
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= need:
            return round((i - need + 1) / fs, 2)
    return None


def _not_analysed(quality: dict | None, reason: str) -> dict:
    return {
        "detected": False, "analysed": False, "reason": reason,
        "tremor_present": False, "tremor_frequency_hz": 0.0, "frequency_reliable": False,
        "tremor_power_ratio": 0.0, "amplitude_px": 0.0, "amplitude_norm": 0.0,
        "amplitude_cm": None, "updrs_amplitude_grade": None, "hand_size_px": 0.0,
        "rms_velocity": 0.0, "velocity_cv": 0.0, "jitter_index": 0.0,
        "peak_prominence": 0.0, "tremor_onset_s": None, "searched_band_hz": None,
        "stability_score": None, "classification": "Not analysed",
        "quality": quality,
    }


def analyze_hand(frames: list[dict] | None, *, expected_duration_s: float, reported_fps: float | None = None):
    if not frames:
        return _not_analysed(None, "no frames for this hand"), [], 0.0

    t = [f["t"] for f in frames]
    x = [f["x"] for f in frames]
    y = [f["y"] for f in frames]
    s = [f.get("s") if f.get("s") is not None else f.get("z", np.nan) for f in frames]
    san = sanitize(t, x, y, s)
    q = assess(
        san, expected_duration_s=expected_duration_s, min_fps=K.MIN_TREMOR_FPS,
        min_coverage=K.MIN_COVERAGE, max_gap_s=K.MAX_GAP_S, min_samples=45, reported_fps=reported_fps,
    )
    sizes = san.values[2]
    valid_sizes = sizes[np.isfinite(sizes) & (sizes > 5)]
    if len(valid_sizes) < 0.5 * max(1, len(sizes)):
        q.ok = False
        q.issues.append("hand size could not be measured (keep the whole hand in view)")
    if not q.ok:
        return _not_analysed(q.to_dict(), "; ".join(q.issues)), [], 0.0

    hand_size = float(np.median(valid_sizes))
    ts = san.t_s
    xs, ys = san.values[0] / hand_size, san.values[1] / hand_size   # hand-size units

    fs = choose_resample_rate(ts)
    tg, xg = resample_uniform(ts, xs, fs)
    _, yg = resample_uniform(ts, ys, fs)
    xg = highpass(xg, 1.0, fs)
    yg = highpass(yg, 1.0, fs)
    axis = principal_axis(xg, yg)

    f_hi = min(K.TREMOR_BAND[1], max_resolvable_hz(fs))
    freq_reliable = f_hi >= K.PD_TREMOR_BAND[1]
    f, p = welch_psd(axis, fs)
    peak = spectral_peak(f, p, K.TREMOR_BAND[0], f_hi) if f_hi > K.TREMOR_BAND[0] + 0.5 else None

    band_sig = bandpass(axis, K.TREMOR_BAND[0], min(K.TREMOR_BAND[1], 0.45 * fs), fs)
    # Amplitude from the PSD (no filter roll-off): band RMS -> sinusoid peak-to-peak.
    amp_hi = min(K.TREMOR_BAND[1], 0.5 * fs)
    amp_norm = float(2.0 * np.sqrt(2.0) * np.sqrt(band_power(f, p, K.TREMOR_BAND[0], amp_hi)))
    amp_cm = amp_norm * K.REFERENCE_PALM_LENGTH_CM

    prominence = peak.prominence if peak else 0.0
    tremor_present = bool(peak and prominence >= K.TREMOR_PEAK_PROMINENCE and amp_cm >= K.TREMOR_MIN_AMPLITUDE_CM)
    freq = peak.freq_hz if (tremor_present and freq_reliable) else 0.0
    total = band_power(f, p, K.TREMOR_BAND[0], f_hi)
    pd_ratio = band_power(f, p, *K.PD_TREMOR_BAND) / total if (tremor_present and total > 0) else 0.0

    grade_c = _continuous_grade(amp_cm) if tremor_present else 0.0
    # grade 1: <1 cm, 2: 1-3 cm, 3: 3-10 cm, 4: >=10 cm (only when a tremor is present)
    grade = min(4, int(np.floor(grade_c)) + 1) if tremor_present else 0
    score = int(round(100 - 25 * grade_c))

    if fs < K.FULL_BAND_TREMOR_FPS:
        q.warnings.append(f"at {fs:.0f} fps only tremor up to {f_hi:.1f} Hz can be detected "
                          f"(faster tremor, e.g. essential tremor, may be missed); "
                          f"{K.FULL_BAND_TREMOR_FPS:.0f}+ fps searches the full {K.TREMOR_BAND[0]:.0f}-{K.TREMOR_BAND[1]:.0f} Hz band")

    vel = np.hypot(np.diff(xg), np.diff(yg)) * fs * hand_size   # px/s
    rms_vel = float(np.sqrt(np.mean(vel ** 2))) if len(vel) else 0.0
    vel_cv = float(np.std(vel) / (np.mean(vel) + 1e-9)) if len(vel) else 0.0
    jitter = float(np.clip(np.std(np.diff(vel)) / (rms_vel + 1e-9), 0, 1)) if len(vel) > 2 else 0.0
    onset = _onset_latency(band_sig, fs, K.TREMOR_MIN_AMPLITUDE_CM / K.REFERENCE_PALM_LENGTH_CM) if tremor_present else None

    spectrum, res = amplitude_spectrum(f, p, fmax=15.0, max_bins=60)
    return {
        "detected": True, "analysed": True, "reason": None,
        "tremor_present": tremor_present,
        "tremor_frequency_hz": round(freq, 2),
        "frequency_reliable": freq_reliable,
        "tremor_power_ratio": round(float(pd_ratio), 4),
        "amplitude_px": round(amp_norm * hand_size, 2),
        "amplitude_norm": round(amp_norm, 4),
        "amplitude_cm": round(amp_cm, 2),
        "updrs_amplitude_grade": grade,
        "hand_size_px": round(hand_size, 1),
        "rms_velocity": round(rms_vel, 2),
        "velocity_cv": round(vel_cv, 4),
        "jitter_index": round(jitter, 4),
        "peak_prominence": round(float(prominence), 2),
        "searched_band_hz": [K.TREMOR_BAND[0], round(f_hi, 2)],
        "tremor_onset_s": onset,
        "stability_score": score,
        "classification": GRADE_LABELS[grade],
        "quality": q.to_dict(),
    }, spectrum, res


def analyze_tremor(left: list[dict] | None, right: list[dict] | None, *, mode: str,
                   expected_duration_s: float, reported_fps: float | None = None) -> dict:
    if not left and not right:
        raise AnalysisError("At least one hand's frames are required.")
    lres, lspec, lfr = analyze_hand(left, expected_duration_s=expected_duration_s, reported_fps=reported_fps)
    rres, rspec, rfr = analyze_hand(right, expected_duration_s=expected_duration_s, reported_fps=reported_fps)

    analysed = [(side, r) for side, r in (("left", lres), ("right", rres)) if r["analysed"]]
    if not analysed:
        issues = [f"{side} hand - {r['reason']}" for side, r in (("left", lres), ("right", rres))
                  if r["reason"] != "no frames for this hand"]
        raise AnalysisError("Recording quality too low for the tremor test: " + "; ".join(issues) + ".", issues)

    if len(analysed) == 2:
        la, ra = lres["amplitude_cm"], rres["amplitude_cm"]
        asym = abs(la - ra) / (la + ra) if (la + ra) > 0 else 0.0
        if not (lres["tremor_present"] or rres["tremor_present"]) or asym < 0.25:
            side = "symmetric"
        else:
            side = "left" if la > ra else "right"
    else:
        asym, side = 0.0, analysed[0][0]

    worst = min(analysed, key=lambda sr: sr[1]["stability_score"])[1]
    return {
        "mode": mode,
        "left": lres,
        "right": rres,
        "asymmetry_index": round(float(asym), 3),
        "dominant_side": side,
        "overall_score": worst["stability_score"],
        "overall_severity": worst["classification"],
        "left_spectrum": lspec,
        "right_spectrum": rspec,
        "freq_resolution": lfr or rfr,
        "algorithm_version": K.ALGORITHM_VERSION,
    }
