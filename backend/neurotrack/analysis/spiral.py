"""
Archimedean spiral drawing analysis.

Input: pointer samples {x, y, t[, stroke]} drawn outward from the guide centre.

What changed vs v1 and why:
  * Duplicate timestamps are merged (v1 turned two events in the same ms into a
    0.1 ms interval and a ~10 000x speed spike that looked like tremor).
  * Pen lifts split the drawing into strokes; pen-up time is not drawing time and
    jumps between strokes are not movement.
  * The spiral is fitted to the person's OWN drawing (r = a + b*theta) instead of
    the guide template, so starting at a different angle or drawing a smaller
    spiral is no longer "error" (v1: a perfect spiral rotated 180 deg had 19 px RMSE).
  * Tremor is measured on the radial wobble around that fit, resampled uniformly
    in time, not on the drawing-speed magnitude (which doubles tremor frequency).
  * New: micrographia index (loops getting progressively tighter, a PD sign).
  * Removed: "rigidity" (cannot be inferred from a drawing) and the slow-speed
    "bradykinesia" penalty (users are told to take their time). Speed is reported
    for information only.
"""

from __future__ import annotations

import numpy as np

from neurotrack.core import constants as K
from neurotrack.core.quality import AnalysisError, QualityReport
from neurotrack.core.signal import (
    amplitude_spectrum,
    band_power,
    bandpass,
    highpass,
    max_resolvable_hz,
    resample_uniform,
    sanitize,
    spectral_peak,
)
from scipy.signal import welch

PEN_UP_GAP_S = 0.25          # a gap this long between samples = pen lifted
MIN_TREMOR_STROKE_S = 2.0    # strokes shorter than this are not used for spectra
DEFAULT_CENTER = (320.0, 160.0)
DEFAULT_GUIDE_RADIUS = 140.0


def _strokes(t_s: np.ndarray, stroke_ids: np.ndarray | None) -> list[slice]:
    breaks = np.diff(t_s) > PEN_UP_GAP_S
    if stroke_ids is not None:
        breaks |= np.diff(stroke_ids) != 0
    idx = np.r_[0, np.flatnonzero(breaks) + 1, len(t_s)]
    return [slice(a, b) for a, b in zip(idx[:-1], idx[1:]) if b - a >= 2]


def _robust_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    mad = np.median(np.abs(res - np.median(res))) * 1.4826 + 1e-9
    keep = np.abs(res) < 3.5 * mad
    if keep.sum() >= 10:
        b, a = np.polyfit(x[keep], y[keep], 1)
    return float(a), float(b)


def analyze_spiral(points: list[dict], *, center: tuple[float, float] | None = None,
                   guide_radius: float | None = None, input_device: str = "unknown") -> dict:
    cx, cy = center or DEFAULT_CENTER
    guide_r = guide_radius or DEFAULT_GUIDE_RADIUS
    has_strokes = all("stroke" in p and p["stroke"] is not None for p in points)
    san = sanitize([p["t"] for p in points], [p["x"] for p in points], [p["y"] for p in points],
                   [p["stroke"] if has_strokes else 0 for p in points])
    t, x, y = san.t_s, san.values[0], san.values[1]
    stroke_ids = np.round(san.values[2]) if has_strokes else None

    issues: list[str] = []
    warnings: list[str] = []
    strokes = _strokes(t, stroke_ids) if len(t) >= 2 else []
    drawing_time = float(sum(t[s][-1] - t[s][0] for s in strokes))
    seg_len = [np.hypot(np.diff(x[s]), np.diff(y[s])).sum() for s in strokes]
    path_len = float(sum(seg_len))
    within = np.concatenate([np.diff(t[s]) for s in strokes]) if strokes else np.array([])
    sample_hz = float(1.0 / np.median(within)) if len(within) else 0.0

    if len(t) < 50:
        issues.append(f"only {len(t)} valid points (need at least 50)")
    if drawing_time < 2.0:
        issues.append(f"drawing lasted {drawing_time:.1f}s (need at least 2s)")
    if path_len < 100:
        issues.append("almost no movement was recorded")

    xc, yc = x - cx, y - cy
    r = np.hypot(xc, yc)
    max_r = float(np.percentile(r, 98)) if len(r) else 0.0
    usable = r >= max(3.0, 0.05 * max_r)       # angle is meaningless right at the centre
    theta = np.unwrap(np.arctan2(yc, xc)) if len(r) else np.array([])
    if usable.sum() >= 10:
        direction = np.sign(theta[usable][-1] - theta[usable][0]) or 1.0
        phi = (theta - theta[usable][0]) * direction
        turns = float((phi[usable].max() - phi[usable].min()) / (2 * np.pi))
    else:
        phi, turns = np.zeros_like(r), 0.0
    if not issues and turns < 1.5:
        issues.append(f"only {turns:.1f} turns drawn (need at least 1.5)")

    q = QualityReport(
        ok=not issues, n_samples=len(t), duration_s=drawing_time, effective_fps=sample_hz,
        coverage=1.0, longest_gap_s=float(np.diff(t).max()) if len(t) > 1 else 0.0,
        max_reliable_freq_hz=max_resolvable_hz(sample_hz), issues=issues, warnings=warnings,
    )
    if san.n_merged:
        warnings.append(f"{san.n_merged} duplicate-timestamp points were merged")
    if len(strokes) > 1:
        warnings.append(f"drawing had {len(strokes)} strokes (pen was lifted)")
    if input_device == "mouse":
        warnings.append("mouse input damps tremor; use a stylus or finger on a touchscreen for valid tremor measures")
    if not q.ok:
        raise AnalysisError("Spiral could not be analysed: " + "; ".join(issues) + ".", issues)

    # ── Shape: fit the person's own spiral ───────────────────────────────────
    pu, ru = phi[usable], r[usable]
    a, b = _robust_linear_fit(pu, ru)
    resid_all = r - (a + b * phi)
    rmse = float(np.sqrt(np.mean(resid_all[usable] ** 2)))
    rmse_rel = rmse / max_r if max_r > 0 else 0.0

    spacings = []
    for k in range(int(np.floor(pu.max() / (2 * np.pi)))):
        m = (pu >= 2 * np.pi * k) & (pu < 2 * np.pi * (k + 1))
        if m.sum() >= 8:
            spacings.append(np.polyfit(pu[m], ru[m], 1)[0] * 2 * np.pi)   # px per loop
    spacings = np.array(spacings)
    if len(spacings) >= 2 and np.mean(spacings) > 0:
        spacing_cv = float(np.std(spacings) / np.mean(spacings))
        micro = float(np.clip((spacings[-1] - spacings[0]) / abs(spacings[0]), -1, 1)) if spacings[0] != 0 else 0.0
    else:
        spacing_cv, micro = 0.0, 0.0
    regularity = float(np.clip(1.0 - spacing_cv, 0.0, 1.0))

    # ── Tremor: radial wobble around the fit, uniform in time ────────────────
    fs = float(min(120.0, max(1.0, round(sample_hz))))
    tremor_reliable = sample_hz >= K.MIN_SPIRAL_SAMPLE_HZ
    if not tremor_reliable:
        warnings.append(f"pointer sampled at {sample_hz:.0f} Hz; tremor measures need >= {K.MIN_SPIRAL_SAMPLE_HZ:.0f} Hz")
    nper = int(round(MIN_TREMOR_STROKE_S * fs))
    psds, weights, band_sigs = [], [], []
    for s in strokes:
        if t[s][-1] - t[s][0] < MIN_TREMOR_STROKE_S:
            continue
        _, rr = resample_uniform(t[s], resid_all[s], fs)
        if len(rr) < nper:
            continue
        rr = highpass(rr, 1.0, fs)
        f, p = welch(rr, fs=fs, nperseg=nper, noverlap=nper // 2, detrend="linear", window="hann")
        psds.append(p)
        weights.append(len(rr))
        band_sigs.append(bandpass(rr, K.TREMOR_BAND[0], min(K.TREMOR_BAND[1], 0.45 * fs), fs))
    if psds:
        p = np.average(np.vstack(psds), axis=0, weights=weights)
        f_hi = min(K.TREMOR_BAND[1], max_resolvable_hz(fs))
        peak = spectral_peak(f, p, K.TREMOR_BAND[0], f_hi)
        band = np.concatenate(band_sigs)
        trem_amp_px = float(2 * np.sqrt(2) * np.sqrt(np.mean(band ** 2)))
        total = band_power(f, p, 1.0, f_hi)
        tremor_ratio = band_power(f, p, K.TREMOR_BAND[0], f_hi) / total if total > 0 else 0.0
        band_db = 10 * np.log10(band_power(f, p, K.TREMOR_BAND[0], f_hi) + 1e-12)
        zcr = float(np.mean(np.diff(np.sign(band)) != 0)) if len(band) > 1 else 0.0
        tremor_present = bool(peak and peak.prominence >= K.TREMOR_PEAK_PROMINENCE
                              and trem_amp_px >= 1.5 and trem_amp_px / max_r >= 0.005)
        tremor_freq = round(peak.freq_hz, 2) if (peak and tremor_present) else 0.0
        spectrum, res = amplitude_spectrum(f, p, fmax=15.0, max_bins=80)
    else:
        tremor_reliable = False
        warnings.append(f"no continuous stroke of >= {MIN_TREMOR_STROKE_S:.0f}s, tremor not measured")
        trem_amp_px = tremor_ratio = zcr = 0.0
        band_db, tremor_present, tremor_freq, spectrum, res = -120.0, False, 0.0, [], 0.0

    # ── Scores (0-100, 100 = healthiest). Uncalibrated, see constants.py. ────
    comp = {
        "deviation": float(np.clip(100 - (rmse_rel - 0.03) / 0.17 * 100, 0, 100)),
        "regularity": regularity * 100,
        "micrographia": float(np.clip(100 + (micro + 0.10) / 0.30 * 100, 0, 100)) if micro < -0.10 else 100.0,
    }
    weights_ = {"deviation": 0.30, "regularity": 0.25, "micrographia": 0.20}
    if tremor_reliable:
        comp["tremor"] = (float(np.clip(100 - (trem_amp_px / max_r - 0.005) / 0.04 * 100, 0, 100))
                          if tremor_present else 100.0)
        weights_["tremor"] = 0.25
    score = K.blend_score(comp, weights_)

    issues_found = []
    if tremor_present:
        issues_found.append("tremor")
    if regularity < 0.6 or rmse_rel > 0.12:
        issues_found.append("irregularity")
    if micro < -0.25:
        issues_found.append("micrographia")
    dominant = "none" if not issues_found else ("mixed" if len(issues_found) > 1 else issues_found[0])

    speed = path_len / drawing_time if drawing_time > 0 else 0.0
    return {
        "overall_score": score,
        "severity": K.label_for(score),
        "dominant_impairment": dominant,
        "drawing_duration_s": round(drawing_time, 2),
        "point_count": len(t),
        "stroke_count": len(strokes),
        "input_device": input_device,
        "biomarkers": {
            "tremor_power_ratio": round(float(tremor_ratio), 4),
            "tremor_band_db": round(float(band_db), 2),
            "tremor_frequency_hz": tremor_freq,
            "tremor_amplitude_px": round(trem_amp_px, 2),
            "tremor_present": tremor_present,
            "tremor_reliable": tremor_reliable,
            "rmse_from_ideal": round(rmse, 2),          # px, vs the person's own best-fit spiral
            "rmse_relative": round(rmse_rel, 4),
            "spiral_regularity": round(regularity, 4),
            "radius_cv": round(spacing_cv, 4),          # CV of loop spacing
            "micrographia_index": round(micro, 4),      # <0 = loops getting tighter
            "turns": round(turns, 2),
            "size_ratio": round(max_r / guide_r, 3),
            "avg_speed_px_per_s": round(speed, 1),      # informational, not scored
            "zero_crossing_rate": round(zcr, 4),
        },
        "component_scores": {k: round(v, 1) for k, v in comp.items()},
        "fft_spectrum": spectrum,
        "fft_freq_resolution": res,
        "quality": q.to_dict(),
        "algorithm_version": K.ALGORITHM_VERSION,
    }
