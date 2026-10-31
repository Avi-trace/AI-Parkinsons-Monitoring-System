"""Runtime settings, read from environment variables (see backend/.env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _csv(name: str, default: str = "") -> list[str]:
    return [v.strip() for v in os.environ.get(name, default).split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: os.environ.get("NEUROTRACK_ENV", "development").lower())
    db_path: str = field(default_factory=lambda: os.environ.get(
        "NEUROTRACK_DB_PATH", str(BACKEND_DIR / "data" / "neurotrack.db")))
    api_keys: tuple[str, ...] = field(default_factory=lambda: tuple(_csv("NEUROTRACK_API_KEYS")))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(_csv(
        "NEUROTRACK_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")))
    max_body_bytes: int = field(default_factory=lambda: int(os.environ.get("NEUROTRACK_MAX_BODY_BYTES", 5_000_000)))
    log_level: str = field(default_factory=lambda: os.environ.get("NEUROTRACK_LOG_LEVEL", "INFO").upper())

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def validate(self) -> None:
        if self.is_production and not self.api_keys:
            raise RuntimeError("NEUROTRACK_API_KEYS must be set when NEUROTRACK_ENV=production")
        if self.is_production and "*" in self.cors_origins:
            raise RuntimeError("NEUROTRACK_CORS_ORIGINS must not be '*' in production")
