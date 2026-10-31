"""
NeuroTrack AI — FastAPI backend.

Run (from backend/):
    uvicorn neurotrack.main:app --reload --port 5000
Docs (development only): http://localhost:5000/docs
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from neurotrack.api.routes import api, public
from neurotrack.config import Settings
from neurotrack.core.constants import ALGORITHM_VERSION
from neurotrack.core.quality import AnalysisError
from neurotrack.db.repository import ConflictError, Repository

log = logging.getLogger("neurotrack")


class BodySizeLimit:
    """Pure-ASGI guard: rejects bodies over max_bytes, including chunked uploads
    without a Content-Length header, before they are buffered in memory."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            return await self._reject(send)
        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _TooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _TooLarge:
            await self._reject(send)

    async def _reject(self, send):
        await JSONResponse({"error": f"Request body larger than {self.max_bytes} bytes"}, status_code=413)(
            {"type": "http"}, None, send)


class _TooLarge(Exception):
    pass


def create_app(settings: Settings | None = None, repo: Repository | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.validate()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not settings.api_keys:
        log.warning("NEUROTRACK_API_KEYS is not set: API is open (development mode only)")

    app = FastAPI(
        title="NeuroTrack AI",
        version=ALGORITHM_VERSION,
        description="Research motor-screening API. Not a medical device; results are not a diagnosis.",
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.settings = settings
    app.state.repo = repo or Repository(settings.db_path)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(AnalysisError)
    async def analysis_error(request: Request, exc: AnalysisError):
        return JSONResponse({"error": exc.message, "issues": exc.issues}, status_code=422)

    @app.exception_handler(ConflictError)
    async def conflict(request: Request, exc: ConflictError):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body", "msg": e["msg"]}
                   for e in exc.errors()][:20]
        first = details[0] if details else {"field": "body", "msg": "invalid"}
        return JSONResponse({"error": f"Invalid request: {first['field']} - {first['msg']}", "details": details},
                            status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "-")
        log.exception("unhandled error (request %s)", rid)
        return JSONResponse({"error": "Internal server error", "request_id": rid}, status_code=500)

    app.include_router(public)
    app.include_router(api)
    app.add_middleware(BodySizeLimit, max_bytes=settings.max_body_bytes)
    return app


app = create_app()
