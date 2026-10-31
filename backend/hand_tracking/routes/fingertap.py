from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from hand_tracking.models.schema import FingerTapRequest, FingerTapResult
from hand_tracking.utils.analyser import analyze_aperture

fingertap_bp = Blueprint("fingertap", __name__)


@fingertap_bp.route("/api/finger-tap", methods=["POST"])
def finger_tap():
    try:
        body = FingerTapRequest.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return jsonify({
            "error": "Invalid request",
            "details": [{"field": ".".join(str(l) for l in err["loc"]), "msg": err["msg"]}
                        for err in e.errors()]
        }), 422

    try:
        frames_raw = [{"v": f.v, "t": f.t} for f in body.frames]
        biomarkers, score, severity, dominant, tap_events = analyze_aperture(
            frames_raw, body.duration_s
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

    # Downsample raw frames to max 300 points for the frontend chart
    frames = frames_raw
    if len(frames) > 300:
        step = len(frames) // 300
        frames = frames[::step]

    result = FingerTapResult(
        patient_id       = body.patient_id,
        hand             = body.hand,
        duration_s       = body.duration_s,
        tap_count        = len(tap_events),
        overall_score    = score,
        severity         = severity,
        dominant_feature = dominant,
        biomarkers       = biomarkers,
        tap_events       = tap_events,
        raw_frames       = frames,
    )
    return jsonify(result.model_dump()), 200


@fingertap_bp.route("/api/finger-tap/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "endpoint": "finger-tap"}), 200