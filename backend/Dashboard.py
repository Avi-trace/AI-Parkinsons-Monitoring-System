"""
Dashboard routes:
  POST /api/session          — save a completed multi-test session
  GET  /api/session/<id>     — fetch one session
  GET  /api/patient/<id>/history  — fetch patient's session history
"""

from flask import Blueprint, request, jsonify
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from session import save_session, get_session, get_patient_history

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/api/session", methods=["POST"])
def create_session():
    body = request.get_json(force=True) or {}
    patient_id = body.get("patient_id", "anonymous")
    result = save_session(
        patient_id=patient_id,
        finger=body.get("finger"),
        spiral=body.get("spiral"),
        tremor=body.get("tremor"),
    )
    return jsonify(result), 201


@dashboard_bp.route("/api/session/<int:session_id>", methods=["GET"])
def fetch_session(session_id):
    s = get_session(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(s), 200


@dashboard_bp.route("/api/patient/<patient_id>/history", methods=["GET"])
def patient_history(patient_id):
    limit = min(int(request.args.get("limit", 10)), 50)
    return jsonify(get_patient_history(patient_id, limit)), 200