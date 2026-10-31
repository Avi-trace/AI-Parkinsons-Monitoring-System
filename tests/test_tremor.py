import pytest

from neurotrack.analysis.tremor import analyze_tremor
from neurotrack.core.quality import AnalysisError
from synthetic import hand


def right(**kw):
    return analyze_tremor(None, hand(**kw), mode="rest", expected_duration_s=15)["right"]


@pytest.mark.parametrize("freq", [4.0, 5.0, 6.0, 8.0, 10.0])
def test_frequency_is_accurate_at_30fps(freq):
    r = right(freq=freq, fps=30)
    assert r["tremor_present"] and abs(r["tremor_frequency_hz"] - freq) < 0.3


@pytest.mark.parametrize("fps", [20, 30, 60])
def test_amplitude_in_cm_is_accurate(fps):
    r = right(amp_cm=2.0, fps=fps)
    assert abs(r["amplitude_cm"] - 2.0) < 0.3


@pytest.mark.parametrize("amp,grade", [(0.5, 1), (2.0, 2), (5.0, 3), (12.0, 4)])
def test_updrs_amplitude_grades(amp, grade):
    assert right(amp_cm=amp)["updrs_amplitude_grade"] == grade


def test_landmark_jitter_is_not_tremor():
    # v1 reported 1 px of jitter as a 4.4 Hz "Mild" tremor.
    r = right(amp_cm=0.0, jitter_px=1.0)
    assert not r["tremor_present"] and r["classification"] == "Normal"


def test_dropped_frames_still_give_the_right_frequency():
    r = right(drop=0.4)
    assert r["tremor_present"] and abs(r["tremor_frequency_hz"] - 5.0) < 0.3


def test_low_fps_is_rejected_instead_of_aliasing():
    # v1: an 8 Hz tremor tracked at 10 fps was reported as 3 Hz.
    with pytest.raises(AnalysisError, match="fps"):
        right(freq=8.0, fps=10)


def test_between_15_and_30_fps_the_searched_band_is_reported():
    r = right(fps=20)
    assert r["searched_band_hz"][1] == pytest.approx(8.0)
    assert any("may be missed" in w for w in r["quality"]["warnings"])


def test_missing_hand_is_not_analysed_never_normal():
    out = analyze_tremor(None, hand(), mode="postural", expected_duration_s=15)
    assert out["left"]["analysed"] is False
    assert out["left"]["stability_score"] is None and out["left"]["classification"] == "Not analysed"


def test_no_usable_hand_rejects_the_request():
    # v1 returned score 100 "Normal" when no hand was detected.
    with pytest.raises(AnalysisError):
        analyze_tremor(None, None, mode="rest", expected_duration_s=15)
    with pytest.raises(AnalysisError, match="fps"):
        analyze_tremor(hand(fps=6), None, mode="rest", expected_duration_s=15)


def test_overall_follows_the_worse_hand_and_asymmetry():
    out = analyze_tremor(hand(amp_cm=0.0, seed=3), hand(amp_cm=4.0), mode="rest", expected_duration_s=15)
    assert out["overall_score"] == out["right"]["stability_score"]
    assert out["dominant_side"] == "right" and out["asymmetry_index"] > 0.5


def test_legacy_z_field_is_accepted_as_hand_size():
    frames = [{**f, "z": f.pop("s")} for f in hand()]
    r = analyze_tremor(None, frames, mode="postural", expected_duration_s=15)["right"]
    assert r["analysed"] and r["tremor_present"]
