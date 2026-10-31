import math

import numpy as np

from neurotrack.core.quality import assess
from neurotrack.core.signal import resample_uniform, sanitize, spectral_peak, welch_psd
from synthetic import frame_times


def test_sanitize_drops_nan_sorts_and_merges_duplicate_timestamps():
    s = sanitize([30, 10, 10, 20, float("nan")], [3, 1, 3, 2, 9])
    assert s.t_s.tolist() == [0.01, 0.02, 0.03]
    assert s.values[0].tolist() == [2.0, 2.0, 3.0]          # the two t=10 samples averaged
    assert (s.n_dropped, s.n_merged, s.reordered) == (1, 1, True)


def test_resampling_irregular_frames_preserves_tremor_amplitude():
    t = np.array(frame_times(30, 10)) / 1000.0
    x = np.sin(2 * math.pi * 5 * t)
    _, xr = resample_uniform(t, x, 30.0)
    assert 0.95 < np.sqrt(2) * xr[30:-30].std() < 1.05      # linear interp lost ~9 % here


def test_welch_peak_frequency_is_accurate():
    fs = 30.0
    t = np.arange(0, 15, 1 / fs)
    f, p = welch_psd(np.sin(2 * math.pi * 5.3 * t), fs)
    peak = spectral_peak(f, p, 3, 12)
    assert abs(peak.freq_hz - 5.3) < 0.1 and peak.prominence > 100


def test_quality_rejects_low_fps_and_long_gaps():
    t = frame_times(8, 10)
    q = assess(sanitize(t, t), expected_duration_s=10, min_fps=15, min_coverage=0.8, max_gap_s=0.5, min_samples=10)
    assert not q.ok and any("fps" in i for i in q.issues)

    t = [x for x in frame_times(30, 10) if not 3000 < x < 5000]
    q = assess(sanitize(t, t), expected_duration_s=10, min_fps=15, min_coverage=0.8, max_gap_s=0.5, min_samples=10)
    assert not q.ok and any("lost" in i for i in q.issues)
