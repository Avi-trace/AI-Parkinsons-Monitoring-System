import pytest

from neurotrack.db.repository import ConflictError
from synthetic import hand, spiral, tapping


def analysed(client, patient="PT001"):
    tap = client.post("/api/finger-tap", json={"patient_id": patient, "frames": tapping()}).json()
    trem = client.post("/api/tremor-test", json={"patient_id": patient, "mode": "rest",
                                                  "right_hand": hand(amp_cm=4)}).json()
    spi = client.post("/api/spiral-test", json={"patient_id": patient, "points": spiral()}).json()
    return tap, trem, spi


def test_health_endpoints(client):
    assert client.get("/api/health").json()["status"] == "ok"
    for name in ("finger-tap", "tremor-test", "spiral-test"):
        assert client.get(f"/api/{name}/health").status_code == 200


def test_analysis_results_are_stored_and_returned_with_an_id(client):
    tap, trem, spi = analysed(client)
    assert all(isinstance(r["result_id"], int) for r in (tap, trem, spi))
    assert tap["algorithm_version"] and "quality" in tap


def test_session_uses_server_scores_and_worst_result_per_test(client):
    tap, trem, spi = analysed(client)
    # The legacy dashboard posts whole result objects; tampered scores are ignored.
    body = {"patient_id": "PT001", "finger": {**tap, "overall_score": 0}, "tremor": trem, "spiral": spi,
            "medication_state": "on", "minutes_since_dose": 90}
    r = client.post("/api/session", json=body)
    assert r.status_code == 201
    s = r.json()
    assert s["finger_score"] == tap["overall_score"]
    expected = round((tap["overall_score"] + trem["overall_score"] + spi["overall_score"]) / 3)
    assert s["motor_screening_index"] == s["composite"] == expected
    assert s["medication_state"] == "on" and len(s["tests"]) == 3


def test_missing_tests_are_not_counted_as_healthy(client):
    _, trem, _ = analysed(client)
    s = client.post("/api/session", json={"patient_id": "PT001", "tremor": trem}).json()
    assert s["motor_screening_index"] == trem["overall_score"]
    assert s["finger_score"] is None and s["spiral_score"] is None


def test_idempotent_session_creation(client):
    tap, _, _ = analysed(client)
    body = {"patient_id": "PT001", "result_ids": [tap["result_id"]], "idempotency_key": "visit-0001"}
    first, again = client.post("/api/session", json=body), client.post("/api/session", json=body)
    assert first.status_code == 201 and again.status_code == 200
    assert first.json()["id"] == again.json()["id"]


def test_results_cannot_be_reused_or_borrowed(client):
    tap, _, _ = analysed(client)
    client.post("/api/session", json={"patient_id": "PT001", "result_ids": [tap["result_id"]]})
    reused = client.post("/api/session", json={"patient_id": "PT001", "result_ids": [tap["result_id"]]})
    assert reused.status_code == 422 and "already part of session" in reused.json()["error"]
    other, _, _ = analysed(client, patient="PT002")
    borrowed = client.post("/api/session", json={"patient_id": "PT001", "result_ids": [other["result_id"]]})
    assert borrowed.status_code == 422 and "different patient" in borrowed.json()["error"]


def test_legacy_results_without_id_are_rejected(client):
    r = client.post("/api/session", json={"patient_id": "PT001", "finger": {"overall_score": 99}})
    assert r.status_code == 422 and "result_id" in r.json()["error"]


def test_history_is_newest_first_with_legacy_fields(client):
    for _ in range(2):
        tap, _, _ = analysed(client)
        client.post("/api/session", json={"patient_id": "PT001", "result_ids": [tap["result_id"]]})
    hist = client.get("/api/patient/PT001/history?limit=10").json()
    assert len(hist) == 2 and hist[0]["id"] > hist[1]["id"]
    assert {"composite", "risk_level", "finger_score", "created_at"} <= set(hist[0])


@pytest.mark.parametrize("method,url,kwargs", [
    ("post", "/api/session", {"content": "not json", "headers": {"content-type": "text/plain"}}),
    ("post", "/api/finger-tap", {"content": '{"frames":[{"v":NaN,"t":0}]}', "headers": {"content-type": "application/json"}}),
    ("post", "/api/finger-tap", {"json": {"frames": [{"v": 1, "t": 1}] * 25, "patient_id": "../etc"}}),
    ("post", "/api/tremor-test", {"json": {}}),
    ("post", "/api/spiral-test", {"json": {"points": [{"x": 1, "y": 1, "t": 1}] * 60}}),
    ("get", "/api/patient/PT001/history?limit=abc", {}),
    ("get", "/api/patient/PT001/history?limit=500", {}),
])
def test_bad_input_is_a_422_never_a_500(client, method, url, kwargs):
    r = getattr(client, method)(url, **kwargs)
    assert r.status_code == 422 and "error" in r.json()


def test_oversized_bodies_are_rejected(client):
    big = '{"frames":[' + ",".join(['{"v":1,"t":1}'] * 400_000) + "]}"
    assert client.post("/api/finger-tap", content=big, headers={"content-type": "application/json"}).status_code == 413


def test_api_key_is_enforced_when_configured(keyed_client):
    assert keyed_client.get("/api/patient/PT001/history").status_code == 401
    assert keyed_client.get("/api/patient/PT001/history", headers={"X-API-Key": "wrong"}).status_code == 401
    assert keyed_client.get("/api/patient/PT001/history", headers={"X-API-Key": "test-key-123"}).status_code == 200
    assert keyed_client.get("/api/health").status_code == 200


def test_production_requires_keys_and_no_wildcard_cors():
    from neurotrack.config import Settings
    with pytest.raises(RuntimeError):
        Settings(env="production", api_keys=()).validate()
    with pytest.raises(RuntimeError):
        Settings(env="production", api_keys=("k",), cors_origins=("*",)).validate()


def test_concurrent_attach_is_rolled_back(repo):
    rid = repo.add_result(patient_id="P", test_type="spiral", side=None, mode=None, score=90,
                          severity="Normal", algorithm_version="x", result={})
    repo.create_session(patient_id="P", medication_state="unknown", minutes_since_dose=None, device=None,
                        algorithm_version="x", screening_index=90, screening_label="ok",
                        idempotency_key=None, result_ids=[rid])
    with pytest.raises(ConflictError):
        repo.create_session(patient_id="P", medication_state="unknown", minutes_since_dose=None, device=None,
                            algorithm_version="x", screening_index=90, screening_label="ok",
                            idempotency_key=None, result_ids=[rid])
    assert len(repo.patient_history("P", 10)) == 1      # the failed session was not left behind
