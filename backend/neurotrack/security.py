"""API-key check. Interim protection until real clinician/patient login is built
with the frontend rework. Keys come from NEUROTRACK_API_KEYS (comma-separated)."""

from __future__ import annotations

import hmac
import logging

from fastapi import Header, HTTPException, Request, status

log = logging.getLogger("neurotrack.security")


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    settings = request.app.state.settings
    if not settings.api_keys:
        return   # development mode without keys: open (a warning is logged at startup)
    if x_api_key and any(hmac.compare_digest(x_api_key, k) for k in settings.api_keys):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key header")
