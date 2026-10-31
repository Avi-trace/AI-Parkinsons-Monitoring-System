"""HTTP routes. Paths are unchanged from v1 so the current frontend keeps working."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response

from neurotrack import services
from neurotrack.api import schemas as S
from neurotrack.core.constants import ALGORITHM_VERSION
from neurotrack.db.repository import Repository
from neurotrack.security import require_api_key

PatientIdPath = Path(pattern=r"^[A-Za-z0-9_\-]{1,64}$")

public = APIRouter(prefix="/api", tags=["health"])
api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


def repo(request: Request) -> Repository:
    return request.app.state.repo


@public.get("/health")
def health():
    return {"status": "ok", "algorithm_version": ALGORITHM_VERSION}


def _legacy_health(name: str):
    def handler():
        return {"status": "ok", "endpoint": name}
    return handler


for _name in ("finger-tap", "tremor-test", "spiral-test"):   # legacy per-test health checks
    public.add_api_route(f"/{_name}/health", _legacy_health(_name), methods=["GET"])


# ── Analysis ─────────────────────────────────────────────────────────────────
# Plain `def` (not async): numpy/scipy work runs in FastAPI's thread pool and
# never blocks the event loop.
@api.post("/finger-tap", tags=["tests"])
def finger_tap(body: S.FingerTapRequest, r: Repository = Depends(repo)):
    return services.run_finger_tap(body, r)


@api.post("/tremor-test", tags=["tests"])
def tremor_test(body: S.TremorRequest, r: Repository = Depends(repo)):
    return services.run_tremor(body, r)


@api.post("/spiral-test", tags=["tests"])
def spiral_test(body: S.SpiralRequest, r: Repository = Depends(repo)):
    return services.run_spiral(body, r)


# ── Sessions & patients ──────────────────────────────────────────────────────
@api.post("/session", status_code=201, tags=["sessions"])
def create_session(body: S.SessionCreate, response: Response, r: Repository = Depends(repo)):
    session, created = services.create_session(body, r)
    if not created:
        response.status_code = 200
    return session


@api.get("/session/{session_id}", tags=["sessions"])
def get_session(session_id: int, r: Repository = Depends(repo)):
    s = r.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return services.present_session(s)


@api.get("/patient/{patient_id}/history", tags=["sessions"])
def patient_history(patient_id: str = PatientIdPath, limit: int = Query(10, ge=1, le=50),
                    r: Repository = Depends(repo)):
    return [services.present_session(s) for s in r.patient_history(patient_id, limit)]


@api.put("/patient/{patient_id}", tags=["patients"])
def upsert_patient(body: S.PatientUpsert, patient_id: str = PatientIdPath, r: Repository = Depends(repo)):
    return r.upsert_patient(patient_id, body.dominant_hand)


@api.get("/patient/{patient_id}", tags=["patients"])
def get_patient(patient_id: str = PatientIdPath, r: Repository = Depends(repo)):
    p = r.get_patient(patient_id)
    if not p:
        raise HTTPException(404, "Patient not found")
    return p
