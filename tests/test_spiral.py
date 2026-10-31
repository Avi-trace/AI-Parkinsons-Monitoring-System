import math

import pytest

from neurotrack.analysis.spiral import analyze_spiral
from neurotrack.core.quality import AnalysisError
from synthetic import spiral


@pytest.mark.parametrize("kw", [{}, {"start_angle": math.pi}, {"scale": 0.7}, {"dur": 25}])
def test_clean_spirals_are_normal_whatever_their_start_size_or_speed(kw):
    # v1: rotated -> 19 px "error", 70 % size -> 24 px, slow drawing -> "bradykinesia".
    r = analyze_spiral(spiral(**kw))
    assert r["severity"] == "Normal" and r["dominant_impairment"] == "none"
    assert r["biomarkers"]["rmse_from_ideal"] < 2


def test_duplicate_timestamps_do_not_create_fake_tremor():
    r = analyze_spiral(spiral(dup_every=10))
    assert not r["biomarkers"]["tremor_present"] and r["severity"] == "Normal"


def test_tremor_is_detected_with_its_frequency():
    r = analyze_spiral(spiral(tremor_hz=5, tremor_px=4))
    b = r["biomarkers"]
    assert b["tremor_present"] and abs(b["tremor_frequency_hz"] - 5) < 0.3
    assert r["dominant_impairment"] == "tremor" and r["severity"] != "Normal"


def test_micrographia_is_measured():
    r = analyze_spiral(spiral(micro=-0.5))
    assert r["biomarkers"]["micrographia_index"] < -0.25 and "micrographia" in r["dominant_impairment"]


def test_no_rigidity_or_speed_based_bradykinesia_labels():
    labels = {analyze_spiral(spiral(**kw))["dominant_impairment"] for kw in ({}, {"dur": 40}, {"micro": -0.5})}
    assert not labels & {"rigidity", "bradykinesia"}


def test_static_or_empty_drawings_are_rejected():
    # v1 returned "Normal" for 40 identical points.
    with pytest.raises(AnalysisError):
        analyze_spiral([{"x": 320, "y": 160, "t": 1000 + i} for i in range(60)])


def test_pen_lift_splits_strokes_and_pen_up_time_is_excluded():
    pts = spiral(dur=8)
    half = len(pts) // 2
    lifted = [dict(p, stroke=0) for p in pts[:half]] + [dict(p, t=p["t"] + 3000, stroke=1) for p in pts[half:]]
    r = analyze_spiral(lifted)
    assert r["stroke_count"] == 2
    assert abs(r["drawing_duration_s"] - 8) < 0.3      # the 3 s pause is not drawing time
    assert r["severity"] == "Normal"


def test_mouse_input_warns():
    r = analyze_spiral(spiral(), input_device="mouse")
    assert any("mouse" in w for w in r["quality"]["warnings"])
