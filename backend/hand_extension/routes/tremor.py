"""
POST /api/tremor-test

Postural tremor test: arms extended forward, hold still for 15s.
Frontend (MediaPipe Hands) sends per-frame wrist landmark positions
for left and/or right hand; this endpoint returns FFT-based tremor scores.
"""

from flask import Blueprint, request, jsonify
from pydantic import ValidationError

from hand_extension.models.schema import TremorTestRequest, TremorTestResult
from hand_extension.utils.analyzer import analyze_both_hands

tremor_bp = Blueprint("tremor", __name__)


@tremor_bp.route("/api/tremor-test", methods=["POST"])
def tremor_test():
    try:
        body = TremorTestRequest.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return jsonify({
            "error": "Invalid request",
            "details": [
                {"field": ".".join(str(l) for l in err["loc"]), "msg": err["msg"]}
                for err in e.errors()
            ]
        }), 422

    if not body.left_hand and not body.right_hand:
        return jsonify({
            "error": "Invalid request",
            "details": [{"field": "left_hand/right_hand", "msg": "At least one hand's data is required"}]
        }), 422

    try:
        import logging
        lf = body.left_hand  or []
        rf = body.right_hand or []
        lr = len(lf); rr = len(rf)
        if lr > 5:
            dur = (lf[-1].t - lf[0].t) / 1000
            sr  = lr / max(dur, 0.1)
            xs  = [f.x for f in lf]
            logging.warning(f"[tremor] L: {lr} frames, {sr:.1f} fps, x-range {min(xs):.1f}-{max(xs):.1f}, hand_size={sum(f.z for f in lf)/lr:.1f}px")
        if rr > 5:
            dur = (rf[-1].t - rf[0].t) / 1000
            sr  = rr / max(dur, 0.1)
            xs  = [f.x for f in rf]
            logging.warning(f"[tremor] R: {rr} frames, {sr:.1f} fps, x-range {min(xs):.1f}-{max(xs):.1f}, hand_size={sum(f.z for f in rf)/rr:.1f}px")
        analysis = analyze_both_hands(body.left_hand, body.right_hand)
    except Exception as e:
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

    result = TremorTestResult(
        patient_id=body.patient_id,
        duration_s=body.duration_s,
        **analysis,
    )

    return jsonify(result.model_dump()), 200


@tremor_bp.route("/api/tremor-test/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "endpoint": "tremor-test"}), 200