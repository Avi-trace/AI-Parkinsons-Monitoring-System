"""
Single source of truth for every band, threshold and version used by the analyzers.

IMPORTANT: the score thresholds below are NOT clinically validated. They are
placeholders chosen to be physiologically plausible. They must be calibrated
against clinician MDS-UPDRS ratings before any clinical claim is made.
Bump ALGORITHM_VERSION whenever an analyzer's output changes, so longitudinal
comparisons never silently mix results from different algorithms.
"""

ALGORITHM_VERSION = "2.0.0"

# ── Frequency bands (Hz) ─────────────────────────────────────────────────────
TREMOR_BAND = (3.0, 12.0)       # pathological tremor search range (PD + ET)
PD_TREMOR_BAND = (4.0, 6.0)     # classic parkinsonian rest-tremor band
VOLUNTARY_BAND = (0.3, 3.0)     # drift / voluntary movement, excluded from tremor

# A frequency f can only be resolved when the sample rate is >= this factor * f.
# (2.0 is the Nyquist limit; 2.5 leaves headroom for the anti-alias roll-off.)
NYQUIST_MARGIN = 2.5

# ── Hand geometry ────────────────────────────────────────────────────────────
# Wrist (landmark 0) -> middle-finger MCP (landmark 9) in an average adult hand.
# Used ONLY to turn hand-size-normalised amplitudes into approximate centimetres.
REFERENCE_PALM_LENGTH_CM = 9.5

# ── Recording quality gates ──────────────────────────────────────────────────
MIN_TREMOR_FPS = 15.0            # below this a tremor test is rejected (can't resolve 6 Hz)
FULL_BAND_TREMOR_FPS = 30.0      # needed to search the whole 3-12 Hz band
MIN_TAPPING_FPS = 20.0           # tapping up to ~7 Hz needs >= 20 fps for clean peaks
MIN_COVERAGE = 0.80              # fraction of the recording the hand must be tracked
MAX_GAP_S = 0.5                  # a single tracking gap longer than this fails quality
MIN_SPIRAL_SAMPLE_HZ = 30.0      # pointer sampling needed for spiral tremor analysis

# ── Tremor detection ─────────────────────────────────────────────────────────
TREMOR_PEAK_PROMINENCE = 5.0     # spectral peak must be >= 5x the median band PSD (~7 dB)
TREMOR_MIN_AMPLITUDE_CM = 0.2    # peak-to-peak; below this is landmark jitter
# MDS-UPDRS 3.15/3.17 amplitude bins (cm): 0 none, 1 <1, 2 1-3, 3 3-10, 4 >=10
UPDRS_AMPLITUDE_BINS_CM = (1.0, 3.0, 10.0)

# ── Severity labels for 0-100 indices (uncalibrated placeholders) ────────────
SCORE_LABELS = ((80, "Normal"), (60, "Mild"), (40, "Moderate"), (0, "Severe"))

SCREENING_LABELS = (
    (80, "Within expected range"),
    (60, "Mild changes"),
    (40, "Moderate changes"),
    (0, "Marked changes"),
)

# Deprecated alias kept so the current dashboard keeps rendering.
LEGACY_RISK_LABELS = ((80, "Low"), (60, "Moderate"), (40, "High"), (0, "Very High"))


def blend_score(components: dict[str, float], weights: dict[str, float]) -> int:
    """0.6 x weighted mean + 0.4 x worst component.

    A plain weighted mean lets one clearly abnormal domain (e.g. a strong
    decrement) be averaged away by four normal ones."""
    total = sum(weights.values())
    mean = sum(components[k] * w for k, w in weights.items()) / total
    worst = min(components[k] for k in weights)
    return int(round(0.6 * mean + 0.4 * worst))


def label_for(score: float | None, table=SCORE_LABELS) -> str:
    if score is None:
        return "Unknown"
    for threshold, label in table:
        if score >= threshold:
            return label
    return table[-1][1]
