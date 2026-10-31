from pydantic import BaseModel, field_validator
from typing import List, Optional


class HandFrame(BaseModel):
    x: float         # composite centroid x (px)
    y: float         # composite centroid y (px)
    z: float = 0.0   # wrist→middleMCP distance (px) — hand size for normalization
    t: float         # timestamp ms from session start


class TremorTestRequest(BaseModel):
    patient_id: str = "anonymous"
    duration_s: float = 15.0
    left_hand:  Optional[List[HandFrame]] = None
    right_hand: Optional[List[HandFrame]] = None

    @field_validator("left_hand", "right_hand")
    @classmethod
    def min_frames(cls, v):
        if v is not None and len(v) < 15:
            raise ValueError("At least 15 frames required per hand")
        return v


class HandTremorScores(BaseModel):
    detected:            bool
    tremor_frequency_hz: float   # dominant freq in 3–12Hz scan band
    tremor_power_ratio:  float   # PD band (4–6Hz) energy fraction
    amplitude_px:        float   # raw peak-to-peak displacement (px)
    amplitude_norm:      float   # amplitude / hand_size (camera-distance-invariant)
    hand_size_px:        float   # wrist→middleMCP px (normalization reference)
    rms_velocity:        float   # RMS frame-to-frame velocity (px/s)
    velocity_cv:         float   # coefficient of variation of velocity
    jitter_index:        float   # acceleration irregularity proxy
    stability_score:     int     # 0–100
    classification:      str     # Normal | Mild | Moderate | Severe


class TremorTestResult(BaseModel):
    patient_id:       str
    duration_s:       float
    left:             HandTremorScores
    right:            HandTremorScores
    asymmetry_index:  float
    overall_severity: str
    overall_score:    int
    dominant_side:    str
    left_spectrum:    List[float]
    right_spectrum:   List[float]
    freq_resolution:  float