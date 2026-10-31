from pydantic import BaseModel, field_validator
from typing import List


class Point(BaseModel):
    x: float
    y: float
    t: float  # timestamp in ms


class SpiralTestRequest(BaseModel):
    points: List[Point]
    patient_id: str = "anonymous"
    hand: str = "dominant"  # "dominant" | "non_dominant"

    @field_validator("points")
    @classmethod
    def minimum_points(cls, v):
        if len(v) < 30:
            raise ValueError("At least 30 points required for analysis")
        return v

    @field_validator("hand")
    @classmethod
    def valid_hand(cls, v):
        if v not in ("dominant", "non_dominant"):
            raise ValueError("hand must be 'dominant' or 'non_dominant'")
        return v


class BiomarkerScores(BaseModel):
    tremor_power_ratio: float       # 0–1, PD band energy fraction
    tremor_band_db: float           # dB power in 3–6 Hz band
    rmse_from_ideal: float          # px deviation from Archimedes spiral
    spiral_regularity: float        # 0–1, higher = more regular
    avg_speed_px_per_s: float       # bradykinesia proxy
    zero_crossing_rate: float       # tremor count proxy
    radius_cv: float                # coefficient of variation of radius growth


class SpiralTestResult(BaseModel):
    patient_id: str
    overall_score: int              # 0–100 (100 = healthy)
    severity: str                   # Normal | Mild | Moderate | Severe
    biomarkers: BiomarkerScores
    dominant_impairment: str        # tremor | bradykinesia | rigidity | mixed | none
    drawing_duration_s: float
    point_count: int
    fft_spectrum: List[float]       # magnitude array for frontend chart
    fft_freq_resolution: float      # Hz per bin