"""
Spiral test signal processing and severity scoring.

Pipeline:
  raw (x,y,t) points
    → coordinate normalisation
    → velocity signal
    → FFT tremor analysis (3–6 Hz PD band)
    → Archimedes spiral fit  → RMSE
    → regularity (radius growth CV)
    → zero-crossing rate
    → weighted severity score
"""

import math
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
from typing import List, Tuple

from spiral_pd.models.schema import BiomarkerScores, Point


# ── Canvas centre (must match frontend) ─────────────────────────────────────
CANVAS_W = 640
CANVAS_H = 320
CX = CANVAS_W / 2   # 320
CY = CANVAS_H / 2   # 160
MAX_RADIUS = min(CX, CY) - 20   # 140 px  — same as frontend guide spiral
SPIRAL_TURNS = 3.5


# ── Low-pass pre-filter ──────────────────────────────────────────────────────
def _butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Butterworth low-pass filter to remove high-freq noise before FFT."""
    nyq = fs / 2
    if cutoff >= nyq:
        return data
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


# ── Unwrap angles robustly ───────────────────────────────────────────────────
def _unwrap_angles(angles: np.ndarray) -> np.ndarray:
    return np.unwrap(angles)


# ── FFT tremor analysis ──────────────────────────────────────────────────────
def _fft_analysis(velocities: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Returns:
        freqs        – frequency axis (Hz)
        magnitudes   – normalised magnitude spectrum
        pd_band_power – power in 3–6 Hz window
        total_power  – total spectral power
        band_db      – PD band power in dB
    """
    n = len(velocities)
    yf = rfft(velocities * np.hanning(n))
    freqs = rfftfreq(n, d=1.0 / sample_rate)
    magnitudes = np.abs(yf) / n

    pd_mask = (freqs >= 3.0) & (freqs <= 6.0)
    pd_band_power = float(np.sum(magnitudes[pd_mask] ** 2))
    total_power = float(np.sum(magnitudes[1:] ** 2))  # skip DC

    band_db = 10 * math.log10(pd_band_power + 1e-12)

    return freqs, magnitudes, pd_band_power, total_power, band_db


# ── Zero-crossing rate ───────────────────────────────────────────────────────
def _zero_crossing_rate(signal: np.ndarray) -> float:
    """Fraction of samples where sign changes — proxy for tremor count."""
    crossings = np.sum(np.diff(np.sign(signal - signal.mean())) != 0)
    return float(crossings / max(len(signal) - 1, 1))


# ── Archimedes spiral fit ────────────────────────────────────────────────────
def _rmse_from_ideal(
    xs: np.ndarray, ys: np.ndarray, unwrapped_angles: np.ndarray
) -> float:
    """
    Ideal Archimedes spiral: r(θ) = (|θ| / θ_max) * MAX_RADIUS
    Returns RMSE in pixels between drawn radii and ideal radii.
    """
    theta_max = SPIRAL_TURNS * 2 * math.pi
    abs_angles = np.abs(unwrapped_angles)
    ideal_radii = (abs_angles / theta_max) * MAX_RADIUS
    drawn_radii = np.hypot(xs, ys)
    residuals = drawn_radii - ideal_radii
    return float(np.sqrt(np.mean(residuals ** 2)))


# ── Radius growth regularity ─────────────────────────────────────────────────
def _spiral_regularity(xs: np.ndarray, ys: np.ndarray, n_segments: int = 8) -> Tuple[float, float]:
    """
    Split the path into n_segments. Compute mean radius per segment.
    Regularity = 1 − CV(Δr across segments).
    Also returns radius coefficient of variation (radius_cv).
    """
    radii = np.hypot(xs, ys)
    seg_size = len(radii) // n_segments
    if seg_size < 2:
        return 0.5, 1.0

    seg_means = [radii[i * seg_size:(i + 1) * seg_size].mean() for i in range(n_segments)]
    deltas = np.diff(seg_means)
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)

    cv = std_delta / (abs(mean_delta) + 1e-6)
    regularity = max(0.0, 1.0 - cv / 2.0)

    radius_cv = float(np.std(radii) / (np.mean(radii) + 1e-6))
    return float(np.clip(regularity, 0.0, 1.0)), radius_cv


# ── Speed profile ────────────────────────────────────────────────────────────
def _speed_profile(points: List[Point]) -> Tuple[float, np.ndarray]:
    dists = []
    for i in range(1, len(points)):
        dx = points[i].x - points[i-1].x
        dy = points[i].y - points[i-1].y
        dt = max((points[i].t - points[i-1].t) / 1000.0, 1e-4)
        dists.append(math.hypot(dx, dy) / dt)
    velocities = np.array(dists, dtype=float)
    total_dist = sum(math.hypot(points[i].x - points[i-1].x,
                                points[i].y - points[i-1].y)
                     for i in range(1, len(points)))
    duration = max((points[-1].t - points[0].t) / 1000.0, 0.1)
    avg_speed = total_dist / duration
    return float(avg_speed), velocities


# ── Severity scoring ─────────────────────────────────────────────────────────
WEIGHTS = {
    "tremor":      0.40,
    "rmse":        0.25,
    "regularity":  0.20,
    "speed":       0.15,
}

def _compute_score(
    tremor_ratio: float,
    rmse: float,
    regularity: float,
    avg_speed: float,
) -> int:
    """
    Each component mapped 0–100 (100 = healthiest), then weighted sum.
    """
    # Tremor: 0 % ratio → 100, 50 %+ ratio → 0
    t_score = max(0.0, 100.0 - tremor_ratio * 200.0)

    # RMSE: 0 px → 100, 60 px+ → 0
    r_score = max(0.0, 100.0 - (rmse / 60.0) * 100.0)

    # Regularity: already 0–1
    reg_score = regularity * 100.0

    # Speed: 40–200 px/s → 100, below 40 → scaled down, above 200 → 90 (too fast = tremor)
    if avg_speed < 10:
        sp_score = 10.0
    elif avg_speed < 40:
        sp_score = 10.0 + (avg_speed - 10.0) / 30.0 * 70.0
    elif avg_speed <= 200:
        sp_score = 80.0 + (avg_speed - 40.0) / 160.0 * 20.0
    else:
        sp_score = 90.0

    score = (
        WEIGHTS["tremor"]     * t_score +
        WEIGHTS["rmse"]       * r_score +
        WEIGHTS["regularity"] * reg_score +
        WEIGHTS["speed"]      * sp_score
    )
    return int(round(min(100, max(0, score))))


def _severity_label(score: int) -> str:
    if score >= 80: return "Normal"
    if score >= 60: return "Mild"
    if score >= 40: return "Moderate"
    return "Severe"


def _dominant_impairment(tremor_ratio: float, rmse: float, regularity: float, avg_speed: float) -> str:
    """Identify the primary motor impairment type."""
    issues = []
    if tremor_ratio > 0.20:
        issues.append(("tremor",     tremor_ratio * 3))
    if avg_speed < 60:
        issues.append(("bradykinesia", (60 - avg_speed) / 60))
    if regularity < 0.55:
        issues.append(("rigidity",   1 - regularity))
    if not issues:
        return "none"
    if len(issues) >= 2:
        return "mixed"
    return issues[0][0]


# ── Main entry point ─────────────────────────────────────────────────────────
def analyze(points: List[Point]) -> Tuple[BiomarkerScores, int, str, str, List[float], float]:
    """
    Full analysis pipeline.

    Returns:
        biomarkers, overall_score, severity, dominant_impairment,
        fft_spectrum (list), fft_freq_resolution (float)
    """
    # 1. Extract arrays, centre on canvas
    xs = np.array([p.x - CX for p in points], dtype=float)
    ys = np.array([p.y - CY for p in points], dtype=float)

    # 2. Speed + velocity signal
    avg_speed, velocities = _speed_profile(points)
    duration = (points[-1].t - points[0].t) / 1000.0
    sample_rate = len(points) / max(duration, 0.1)

    # 3. Pre-filter (keep up to 15 Hz, remove sensor noise)
    vel_filtered = _butter_lowpass(velocities, cutoff=15.0, fs=sample_rate)

    # 4. FFT tremor analysis
    freqs, magnitudes, pd_band_power, total_power, band_db = _fft_analysis(vel_filtered, sample_rate)
    tremor_ratio = pd_band_power / max(total_power, 1e-9)

    # 5. Unwrap angles → Archimedes RMSE
    angles = np.arctan2(ys, xs)
    unwrapped = _unwrap_angles(angles)
    rmse = _rmse_from_ideal(xs, ys, unwrapped)

    # 6. Regularity + radius CV
    regularity, radius_cv = _spiral_regularity(xs, ys)

    # 7. Zero-crossing rate on velocity signal
    zcr = _zero_crossing_rate(vel_filtered)

    # 8. Score + labels
    score = _compute_score(tremor_ratio, rmse, regularity, avg_speed)
    severity = _severity_label(score)
    impairment = _dominant_impairment(tremor_ratio, rmse, regularity, avg_speed)

    biomarkers = BiomarkerScores(
        tremor_power_ratio=round(float(tremor_ratio), 4),
        tremor_band_db=round(float(band_db), 2),
        rmse_from_ideal=round(float(rmse), 2),
        spiral_regularity=round(float(regularity), 4),
        avg_speed_px_per_s=round(float(avg_speed), 1),
        zero_crossing_rate=round(float(zcr), 4),
        radius_cv=round(float(radius_cv), 4),
    )

    # Return only first 80 FFT bins for the frontend chart
    spectrum = [round(float(m), 6) for m in magnitudes[:80]]
    freq_res = round(float(freqs[1]) if len(freqs) > 1 else 0.0, 4)

    return biomarkers, score, severity, impairment, spectrum, freq_res