"""
Postural tremor test — signal processing.

Sample rate considerations:
  - Nyquist for 6Hz PD tremor: need >= 12fps minimum, ideally 20+
  - At 8fps (old): aliasing corrupts 4-6Hz band
  - At 20fps (new 320x240 + lite model): clean detection up to 10Hz
  - Analyzer adapts filter cutoffs dynamically to actual sample rate

Normalization:
  - Amplitude normalized by hand_size (wrist→middleMCP px distance)
  - Makes results independent of camera distance

Velocity:
  - RMS velocity (more robust than mean — not biased by outliers)
  - Velocity CV (irregularity of motion)
  - Both computed from frame-to-frame displacements

FIX (see comments marked "FIX:" below):
  Previously, if the effective sample rate was too low to trust the
  frequency-domain (FFT) tremor-frequency estimate, the analyzer bailed
  out entirely via _empty() — discarding amplitude, RMS velocity, jitter,
  and stability score too, even though none of those need a high sample
  rate to be valid. This made the tool report "Not detected" for hands
  that were clearly moving (e.g. a deliberate shake), whenever per-hand
  fps dipped below ~9 (which happens easily during fast motion, since
  MediaPipe re-detection after motion blur is slower than steady tracking).

  Now the analyzer only skips the frequency-specific numbers
  (tremor_frequency_hz, tremor_power_ratio) when the sample rate can't
  support them, and still returns amplitude/velocity/stability from the
  time-domain data, which remains valid at any sample rate >= 15 frames.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
from typing import List, Optional, Tuple
from hand_extension.models.schema import HandFrame, HandTremorScores

SCAN_LO, SCAN_HI  = 3.0, 12.0
PD_LO,   PD_HI    = 4.0,  6.0
NOISE_DB          = 1.0

# Normalized thresholds (fraction of hand size)
NORM_NOISE_FLOOR  = 0.02   # < 2% hand size = noise
NORM_TREMOR_START = 0.05   # > 5% hand size = definite tremor

# Minimum effective Nyquist headroom above PD_LO needed to trust the
# frequency-domain estimate. Below this, we still analyze amplitude/
# velocity/stability, we just don't report a frequency.
MIN_FREQ_HEADROOM = PD_LO  # scan_hi_eff must be >= this


def _hp(data, cutoff, fs, order=2):
    nyq = fs / 2
    c   = np.clip(cutoff / nyq, 1e-3, 0.99)
    if len(data) < order * 6: return data
    b, a = butter(order, c, btype="high")
    return filtfilt(b, a, data)


def _lp(data, cutoff, fs, order=2):
    nyq = fs / 2
    c   = np.clip(cutoff / nyq, 1e-3, 0.99)
    if len(data) < order * 6: return data
    b, a = butter(order, c, btype="low")
    return filtfilt(b, a, data)


def _classify(score: int) -> str:
    if score >= 80: return "Normal"
    if score >= 60: return "Mild"
    if score >= 40: return "Moderate"
    return "Severe"


def _empty() -> HandTremorScores:
    return HandTremorScores(
        detected=False,
        tremor_frequency_hz=0.0,
        tremor_power_ratio=0.0,
        amplitude_px=0.0,
        amplitude_norm=0.0,
        hand_size_px=0.0,
        rms_velocity=0.0,
        velocity_cv=0.0,
        jitter_index=0.0,
        stability_score=100,
        classification="Normal",
    )


def analyze_hand(
    frames: Optional[List[HandFrame]]
) -> Tuple[HandTremorScores, List[float], float]:

    if not frames or len(frames) < 15:
        return _empty(), [], 0.0

    xs = np.array([f.x for f in frames], dtype=float)
    ys = np.array([f.y for f in frames], dtype=float)
    ts = np.array([f.t for f in frames], dtype=float)
    hs = np.array([f.z for f in frames], dtype=float)  # hand size px

    duration_s  = max((ts[-1] - ts[0]) / 1000.0, 0.1)
    sample_rate = len(frames) / duration_s

    # ── FIX: don't bail out entirely on low sample rate ───────────────────
    # Previously: `if scan_hi_eff < PD_LO: return _empty()`, which threw
    # away amplitude/velocity/stability too. Now we just mark the
    # frequency-domain result as unreliable and keep going.
    nyquist       = max(sample_rate / 2, 0.5)
    scan_hi_eff   = min(SCAN_HI, nyquist - 0.5) if nyquist > 0.5 else 0.0
    freq_reliable = scan_hi_eff >= MIN_FREQ_HEADROOM

    # ── Hand size normalization ───────────────────────────────────────────
    valid_hs  = hs[hs > 5]
    hand_size = float(np.median(valid_hs)) if len(valid_hs) > 5 else 60.0

    # ── Centre ────────────────────────────────────────────────────────────
    xs_c = xs - xs.mean()
    ys_c = ys - ys.mean()

    # ── Filter — cutoffs adapt to actual sample rate ──────────────────────
    # Guard lp_cut/hp cutoff against degenerate values at very low fps.
    lp_cut = min(10.0, max(nyquist - 0.5, 1.0))   # low-pass just above scan band
    xs_f   = _lp(_hp(xs_c, 0.5, sample_rate), lp_cut, sample_rate)
    ys_f   = _lp(_hp(ys_c, 0.5, sample_rate), lp_cut, sample_rate)

    # ── Amplitude (normalized) — valid at any sample rate ─────────────────
    disp          = np.hypot(xs_f, ys_f)
    amplitude_px  = float(np.percentile(disp, 95) - np.percentile(disp, 5))
    amplitude_norm = amplitude_px / hand_size

    # ── RMS velocity — valid at any sample rate ───────────────────────────
    # Much more robust than mean: squares emphasise large displacements,
    # root brings back to px/s units. Not biased by outlier-free stretches.
    dt_s       = np.diff(ts) / 1000.0
    dt_s       = np.maximum(dt_s, 1e-3)    # guard against duplicate timestamps
    inst_vel   = np.hypot(np.diff(xs_f), np.diff(ys_f)) / dt_s  # px/s per frame
    rms_vel    = float(np.sqrt(np.mean(inst_vel ** 2)))
    vel_mean   = float(np.mean(inst_vel))
    velocity_cv = float(np.std(inst_vel) / (vel_mean + 1e-6))

    # ── FFT (x and y separately — avoids frequency doubling) ─────────────
    n      = len(xs_f)
    window = np.hanning(n)
    mags_x = np.abs(rfft((xs_f - xs_f.mean()) * window)) / n
    mags_y = np.abs(rfft((ys_f - ys_f.mean()) * window)) / n
    freqs  = rfftfreq(n, d=1.0 / sample_rate)
    mags   = np.sqrt(mags_x**2 + mags_y**2)

    # ── Dominant frequency — only trusted when sample rate supports it ───
    scan_mask = (freqs >= SCAN_LO) & (freqs <= scan_hi_eff) if freq_reliable else np.zeros_like(freqs, dtype=bool)
    dominant_freq   = 0.0
    tremor_detected = False

    if freq_reliable and scan_mask.any() and amplitude_norm >= NORM_NOISE_FLOOR:
        scan_mags  = mags[scan_mask]
        scan_freqs = freqs[scan_mask]
        peak_idx   = np.argmax(scan_mags)
        peak_mag   = scan_mags[peak_idx]
        non_scan   = mags[~scan_mask & (freqs > 0)]
        noise_ref  = np.mean(non_scan) + 1e-12 if len(non_scan) else 1e-12
        db_above   = 20 * np.log10(peak_mag / noise_ref)
        print(f"[analyzer] peak {scan_freqs[peak_idx]:.2f}Hz, {db_above:.1f}dB above noise")
        print(f"frame_size := {len(frames)}, duration_s := {duration_s:.2f}s, sample_rate := {sample_rate:.1f}fps")
        if db_above >= NOISE_DB:
            dominant_freq   = float(scan_freqs[peak_idx])
            tremor_detected = True
    elif not freq_reliable:
        print(f"[analyzer] sample_rate={sample_rate:.1f}fps too low for reliable frequency estimate "
              f"(need ~9+fps) — reporting amplitude/velocity/stability only, no Hz reading")

    # ── PD band power ratio — only meaningful if frequency is reliable ───
    pd_mask      = (freqs >= PD_LO) & (freqs <= PD_HI) if freq_reliable else np.zeros_like(freqs, dtype=bool)
    pd_power     = float(np.sum(mags[pd_mask] ** 2)) if pd_mask.any() else 0.0
    total_power  = float(np.sum(mags[freqs > 0] ** 2)) + 1e-9
    tremor_ratio = min(1.0, pd_power / total_power) if tremor_detected else 0.0

    # ── Jitter (acceleration proxy from velocity differences) ─────────────
    jitter = float(np.clip(np.std(np.diff(inst_vel)) / (rms_vel + 1e-6), 0, 1))

    # ── Stability score ───────────────────────────────────────────────────
    # Note: amp_pen and vel_pen are always available (time-domain only).
    # freq_pen is 0 when frequency isn't reliable/detected, which is the
    # conservative choice — we don't penalize for a signal we can't measure,
    # but amplitude/velocity still drive the score down for genuine shaking.
    norm_above = max(0.0, amplitude_norm - NORM_NOISE_FLOOR)
    amp_pen    = min(35.0, norm_above / (NORM_TREMOR_START - NORM_NOISE_FLOOR) * 25.0)
    freq_pen   = tremor_ratio * 80.0 if tremor_detected else 0.0
    # RMS velocity penalty: normalize by hand_size to make scale-invariant
    rms_norm   = rms_vel / (hand_size + 1e-6)   # RMS velocity as fraction of hand/s
    vel_pen    = min(20.0, rms_norm * 200.0)

    score = int(round(np.clip(
        100.0 - (freq_pen * 0.45 + amp_pen * 0.35 + vel_pen * 0.20),
        0, 100
    )))

    return HandTremorScores(
        detected=True,
        tremor_frequency_hz=round(dominant_freq, 2),
        tremor_power_ratio=round(tremor_ratio, 4),
        amplitude_px=round(amplitude_px, 2),
        amplitude_norm=round(amplitude_norm, 4),
        hand_size_px=round(hand_size, 1),
        rms_velocity=round(rms_vel, 2),
        velocity_cv=round(velocity_cv, 4),
        jitter_index=round(jitter, 4),
        stability_score=score,
        classification=_classify(score),
    ), [round(float(m), 6) for m in mags[:60]], round(float(freqs[1]) if len(freqs) > 1 else 0.0, 4)


def analyze_both_hands(left_frames, right_frames):
    l, ls, frl = analyze_hand(left_frames)
    r, rs, frr = analyze_hand(right_frames)
    fr         = frl or frr

    if l.detected and r.detected:
        diff  = abs(l.stability_score - r.stability_score)
        side  = "symmetric" if diff < 10 else ("left" if l.stability_score < r.stability_score else "right")
        score = int(round((l.stability_score + r.stability_score) / 2))
        asym  = round(diff / 100.0, 3)
    elif l.detected:
        asym, side, score = 0.0, "left",  l.stability_score
    elif r.detected:
        asym, side, score = 0.0, "right", r.stability_score
    else:
        asym, side, score = 0.0, "none",  100

    return dict(
        left=l, right=r,
        asymmetry_index=asym, dominant_side=side,
        overall_score=score, overall_severity=_classify(score),
        left_spectrum=ls, right_spectrum=rs, freq_resolution=fr,
    )