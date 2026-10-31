from pydantic import BaseModel, field_validator
from typing import List


class ApertureFrame(BaseModel):
    """Single frame of thumb-index aperture from MediaPipe."""
    v: float    # aperture in pixels (thumb tip ↔ index tip distance)
    t: float    # timestamp in ms from session start


class FingerTapRequest(BaseModel):
    patient_id:  str = "anonymous"
    hand:        str = "dominant"
    duration_s:  float = 10.0
    frames:      List[ApertureFrame]   # raw aperture stream, NOT discrete taps
    sample_rate: float = 30.0

    @field_validator("frames")
    @classmethod
    def min_frames(cls, v):
        if len(v) < 20:
            raise ValueError("At least 20 frames required")
        return v

    @field_validator("hand")
    @classmethod
    def valid_hand(cls, v):
        if v not in ("dominant", "non_dominant"):
            raise ValueError("hand must be 'dominant' or 'non_dominant'")
        return v


class FingerTapBiomarkers(BaseModel):
    tap_rate:          float   # taps per second
    mean_amplitude:    float   # mean peak aperture (px)
    mean_speed:        float   # mean closing speed (px/s)
    amplitude_decay:   float   # normalised slope of amplitude over time (-1..0)
    speed_decay:       float   # normalised slope of speed over time (-1..0)
    rhythm_cv:         float   # coefficient of variation of inter-tap intervals
    hesitation_count:  int     # gaps > 2.5× mean ITI
    arrest_ratio:      float   # fraction of duration spent in hesitations
    decrement_index:   float   # composite impairment (0–1, higher = worse)


class FingerTapResult(BaseModel):
    patient_id:       str
    hand:             str
    duration_s:       float
    tap_count:        int
    overall_score:    int
    severity:         str
    dominant_feature: str
    biomarkers:       FingerTapBiomarkers
    tap_events:       List[dict]   # [{t, amp, speed}] for frontend chart
    raw_frames:       List[dict]   # [{v, t}] downsampled aperture for signal chart