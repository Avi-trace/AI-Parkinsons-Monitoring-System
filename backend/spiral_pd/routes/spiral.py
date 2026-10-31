"""
POST /api/spiral-test

Accepts a JSON body with (x, y, t) point arrays from the frontend
spiral drawing widget and returns a full severity analysis.
"""

from flask import Blueprint, request, jsonify
from pydantic import ValidationError

from spiral_pd.models.schema import SpiralTestRequest, SpiralTestResult
from spiral_pd.utils.analyser import analyze

spiral_bp = Blueprint("spiral", __name__)


@spiral_bp.route("/api/spiral-test", methods=["POST"])
def spiral_test():
    # ── 1. Parse & validate input ────────────────────────────────────────────
    try:
        body = SpiralTestRequest.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return jsonify({
            "error": "Invalid request",
            "details": [
                {"field": ".".join(str(l) for l in err["loc"]), "msg": err["msg"]}
                for err in e.errors()
            ]
        }), 422

    # ── 2. Run analysis pipeline ─────────────────────────────────────────────
    try:
        biomarkers, score, severity, impairment, spectrum, freq_res = analyze(body.points)
    except Exception as e:
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

    # ── 3. Build response ────────────────────────────────────────────────────
    result = SpiralTestResult(
        patient_id=body.patient_id,
        overall_score=score,
        severity=severity,
        biomarkers=biomarkers,
        dominant_impairment=impairment,
        drawing_duration_s=round(
            (body.points[-1].t - body.points[0].t) / 1000.0, 2
        ),
        point_count=len(body.points),
        fft_spectrum=spectrum,
        fft_freq_resolution=freq_res,
    )

    return jsonify(result.model_dump()), 200


@spiral_bp.route("/api/spiral-test/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "endpoint": "spiral-test"}), 200