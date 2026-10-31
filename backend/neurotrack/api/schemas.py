"""
Request models. Every number is finite and range-checked, every list is size-capped,
so malformed or hostile input becomes a 422 — never a 500 or an out-of-memory.
Field names stay compatible with the current frontend; new fields are optional.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

PatientId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{1,64}$")]
Coord = Annotated[float, Field(ge=-20_000, le=20_000, allow_inf_nan=False)]
Size = Annotated[float, Field(ge=0, le=20_000, allow_inf_nan=False)]
TimeMs = Annotated[float, Field(ge=0, le=3_600_000, allow_inf_nan=False)]   # within 1 h of start
Fps = Annotated[float, Field(gt=0, le=1000, allow_inf_nan=False)]
Hand = Literal["dominant", "non_dominant"]
Side = Literal["left", "right"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ── Finger tap ───────────────────────────────────────────────────────────────
class ApertureFrame(_Model):
    v: Size                       # thumb-tip <-> index-tip distance (px)
    t: TimeMs                     # ms since recording start
    s: Optional[Size] = None      # hand size, wrist -> middle MCP (px); enables normalisation


class FingerTapRequest(_Model):
    patient_id: PatientId = "anonymous"
    hand: Hand = "dominant"
    side: Optional[Side] = None
    duration_s: float = Field(10.0, ge=3, le=60, allow_inf_nan=False)
    frames: list[ApertureFrame] = Field(min_length=20, max_length=10_000)
    reported_fps: Optional[Fps] = None
    sample_rate: Optional[float] = None   # legacy, ignored (rate is measured from timestamps)


# ── Tremor ───────────────────────────────────────────────────────────────────
class HandFrame(_Model):
    x: Coord                      # composite hand centroid (px)
    y: Coord
    t: TimeMs
    s: Optional[Size] = None      # hand size (px)
    z: Optional[Size] = None      # legacy name for hand size


class TremorRequest(_Model):
    patient_id: PatientId = "anonymous"
    mode: Literal["rest", "postural"] = "postural"
    duration_s: float = Field(15.0, ge=5, le=120, allow_inf_nan=False)
    left_hand: Optional[list[HandFrame]] = Field(None, max_length=20_000)
    right_hand: Optional[list[HandFrame]] = Field(None, max_length=20_000)
    reported_fps: Optional[Fps] = None


# ── Spiral ───────────────────────────────────────────────────────────────────
class SpiralPoint(_Model):
    x: Coord
    y: Coord
    t: TimeMs
    stroke: Optional[int] = Field(None, ge=0, le=10_000)
    p: Optional[float] = Field(None, ge=0, le=1, allow_inf_nan=False)    # pen pressure if available


class SpiralRequest(_Model):
    patient_id: PatientId = "anonymous"
    hand: Hand = "dominant"
    side: Optional[Side] = None
    input_device: Literal["mouse", "touch", "pen", "unknown"] = "unknown"
    points: list[SpiralPoint] = Field(min_length=2, max_length=50_000)
    center_x: Optional[Coord] = None
    center_y: Optional[Coord] = None
    guide_radius: Optional[float] = Field(None, gt=0, le=20_000, allow_inf_nan=False)


# ── Sessions / patients ──────────────────────────────────────────────────────
class _ResultRef(_Model):
    """The current dashboard posts whole result objects; only their result_id is used."""
    result_id: Optional[int] = None


class SessionCreate(_Model):
    patient_id: PatientId = "anonymous"
    medication_state: Literal["on", "off", "unknown"] = "unknown"
    minutes_since_dose: Optional[int] = Field(None, ge=0, le=10_080)
    device: Optional[str] = Field(None, max_length=120)
    idempotency_key: Optional[Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{8,64}$")]] = None
    result_ids: list[int] = Field(default_factory=list, max_length=20)
    finger: Optional[_ResultRef] = None      # legacy shape
    spiral: Optional[_ResultRef] = None
    tremor: Optional[_ResultRef] = None


class PatientUpsert(_Model):
    dominant_hand: Optional[Side] = None
