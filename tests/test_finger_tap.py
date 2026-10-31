import pytest

from neurotrack.analysis.finger_tap import analyze_finger_tap
from neurotrack.core.quality import AnalysisError
from synthetic import tapping


def run(**kw):
    return analyze_finger_tap(tapping(**kw), expected_duration_s=kw.get("dur", 10))


@pytest.mark.parametrize("freq", [2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0])
def test_counts_every_tap_across_the_human_range(freq):
    # v1 found 29 of 45 taps at 4.5 Hz and failed outright at 5 Hz.
    assert abs(run(freq=freq)["tap_count"] - freq * 10) <= 1


@pytest.mark.parametrize("freq", [4.0, 4.5, 5.0, 6.0])
def test_healthy_fast_tapping_is_normal(freq):
    r = run(freq=freq)
    assert r["severity"] == "Normal" and r["dominant_feature"] == "none"


def test_camera_distance_does_not_matter():
    near, far = run(hand_px=120), run(hand_px=30)    # v1: far hand -> "could not detect taps"
    assert near["tap_count"] == far["tap_count"]
    assert abs(near["biomarkers"]["mean_amplitude"] - far["biomarkers"]["mean_amplitude"]) < 0.1


def test_amplitude_decrement_is_detected():
    # v1 scored a 50 % decay as 89 "Normal" with amplitude_decay -0.019.
    r = run(freq=4, decay=0.5)
    assert r["biomarkers"]["amplitude_decay"] < -0.3
    assert r["severity"] != "Normal" and r["dominant_feature"] == "amplitude decrement"


def test_small_amplitude_is_flagged():
    r = run(open_ratio=0.3)
    assert r["dominant_feature"] == "small amplitude" and r["severity"] != "Normal"


def test_hesitations_lower_the_score():
    r = run(freq=4, pauses=((3, 4), (6, 7.2)))
    assert r["biomarkers"]["hesitation_count"] >= 2 and r["overall_score"] < 70


def test_tap_rate_uses_real_timestamps_not_claimed_duration():
    r = analyze_finger_tap(tapping(freq=4, dur=10), expected_duration_s=5)   # client claims 5 s
    assert abs(r["biomarkers"]["tap_rate"] - 4) < 0.2


def test_low_fps_is_rejected_not_scored():
    with pytest.raises(AnalysisError, match="fps"):
        run(fps=12)


def test_no_tapping_is_rejected():
    with pytest.raises(AnalysisError):
        run(open_ratio=0.0)


def test_tap_events_and_frames_share_a_time_axis():
    r = run()
    assert r["raw_frames"][0]["t"] == 0
    assert 0 <= r["tap_events"][0]["t"] <= r["raw_frames"][-1]["t"]


def test_works_without_hand_size_but_warns():
    r = analyze_finger_tap(tapping(with_size=False), expected_duration_s=10)
    assert r["amplitude_unit"] == "px" and r["quality"]["warnings"]
