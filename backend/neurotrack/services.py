"""
Application services: glue between validated requests, analyzers and storage.
Routes stay one-liners so this logic is testable without a web server.
"""

from __future__ import annotations

from neurotrack.analysis.finger_tap import analyze_finger_tap
from neurotrack.analysis.spiral import analyze_spiral
from neurotrack.analysis.tremor import analyze_tremor
from neurotrack.api import schemas as S
from neurotrack.core import constants as K
from neurotrack.core.quality import AnalysisError
from neurotrack.db.repository import Repository


def run_finger_tap(req: S.FingerTapRequest, repo: Repository) -> dict:
    frames = [f.model_dump() for f in req.frames]
    result = analyze_finger_tap(frames, expected_duration_s=req.duration_s, reported_fps=req.reported_fps)
    result.update(patient_id=req.patient_id, hand=req.hand, side=req.side)
    result["result_id"] = repo.add_result(
        patient_id=req.patient_id, test_type="finger_tap", side=req.side, mode=req.hand,
        score=result["overall_score"], severity=result["severity"],
        algorithm_version=K.ALGORITHM_VERSION, result=result)
    return result


def run_tremor(req: S.TremorRequest, repo: Repository) -> dict:
    def frames(lst):
        return [f.model_dump() for f in lst] if lst else None
    result = analyze_tremor(frames(req.left_hand), frames(req.right_hand), mode=req.mode,
                            expected_duration_s=req.duration_s, reported_fps=req.reported_fps)
    result.update(patient_id=req.patient_id, duration_s=req.duration_s)
    result["result_id"] = repo.add_result(
        patient_id=req.patient_id, test_type="tremor", side=None, mode=req.mode,
        score=result["overall_score"], severity=result["overall_severity"],
        algorithm_version=K.ALGORITHM_VERSION, result=result)
    return result


def run_spiral(req: S.SpiralRequest, repo: Repository) -> dict:
    center = (req.center_x, req.center_y) if req.center_x is not None and req.center_y is not None else None
    result = analyze_spiral([p.model_dump() for p in req.points], center=center,
                            guide_radius=req.guide_radius, input_device=req.input_device)
    result.update(patient_id=req.patient_id, hand=req.hand, side=req.side)
    result["result_id"] = repo.add_result(
        patient_id=req.patient_id, test_type="spiral", side=req.side, mode=req.hand,
        score=result["overall_score"], severity=result["severity"],
        algorithm_version=K.ALGORITHM_VERSION, result=result)
    return result


def screening_index(results: list[dict]) -> int | None:
    """Mean over test types of the WORST result per type (e.g. worse hand).
    Missing tests are simply absent — they never count as healthy."""
    per_type: dict[str, int] = {}
    for r in results:
        if r["score"] is None:
            continue
        per_type[r["test_type"]] = min(per_type.get(r["test_type"], 101), r["score"])
    if not per_type:
        return None
    return int(round(sum(per_type.values()) / len(per_type)))


def present_session(s: dict) -> dict:
    """Session view. `composite`, `risk_level` and `*_score` are deprecated aliases
    kept so the current dashboard keeps working until the frontend rework."""
    by_type: dict[str, int] = {}
    for r in s["results"]:
        if r["score"] is not None:
            by_type[r["test_type"]] = min(by_type.get(r["test_type"], 101), r["score"])
    idx = s["screening_index"]
    return {
        "id": s["id"],
        "patient_id": s["patient_id"],
        "created_at": s["created_at"],
        "medication_state": s["medication_state"],
        "minutes_since_dose": s["minutes_since_dose"],
        "device": s["device"],
        "algorithm_version": s["algorithm_version"],
        "motor_screening_index": idx,
        "screening_label": s["screening_label"],
        "tests": [
            {"result_id": r["id"], "test_type": r["test_type"], "side": r["side"], "mode": r["mode"],
             "score": r["score"], "severity": r["severity"], "algorithm_version": r["algorithm_version"],
             "summary": r["summary"]}
            for r in s["results"]
        ],
        "disclaimer": "Research screening index, not a diagnosis. Not clinically validated.",
        # deprecated aliases
        "composite": idx,
        "risk_level": K.label_for(idx, K.LEGACY_RISK_LABELS),
        "finger_score": by_type.get("finger_tap"),
        "spiral_score": by_type.get("spiral"),
        "tremor_score": by_type.get("tremor"),
    }


def create_session(req: S.SessionCreate, repo: Repository) -> tuple[dict, bool]:
    """Returns (session, created). Re-posting the same idempotency_key returns the
    existing session instead of creating a duplicate."""
    if req.idempotency_key:
        existing = repo.find_session_by_key(req.idempotency_key)
        if existing:
            return present_session(existing), False

    ids = list(dict.fromkeys(
        list(req.result_ids) + [ref.result_id for ref in (req.finger, req.spiral, req.tremor)
                                if ref is not None and ref.result_id is not None]))
    legacy_without_id = [name for name, ref in (("finger", req.finger), ("spiral", req.spiral), ("tremor", req.tremor))
                         if ref is not None and ref.result_id is None]
    if legacy_without_id:
        raise AnalysisError("Results must come from this server's analysis endpoints (missing result_id for: "
                            + ", ".join(legacy_without_id) + "). Re-run those tests.")
    if not ids:
        raise AnalysisError("A session needs at least one test result.")

    rows = repo.get_results(ids)
    found = {r["id"] for r in rows}
    problems = [f"result {i} not found" for i in ids if i not in found]
    for r in rows:
        if r["patient_id"] != req.patient_id:
            problems.append(f"result {r['id']} belongs to a different patient")
        elif r["session_id"] is not None:
            problems.append(f"result {r['id']} is already part of session {r['session_id']}")
    if problems:
        raise AnalysisError("Cannot create session: " + "; ".join(problems) + ".", problems)

    idx = screening_index(rows)
    sid = repo.create_session(
        patient_id=req.patient_id, medication_state=req.medication_state,
        minutes_since_dose=req.minutes_since_dose, device=req.device,
        algorithm_version=K.ALGORITHM_VERSION, screening_index=idx,
        screening_label=K.label_for(idx, K.SCREENING_LABELS) if idx is not None else None,
        idempotency_key=req.idempotency_key, result_ids=ids)
    return present_session(repo.get_session(sid)), True
