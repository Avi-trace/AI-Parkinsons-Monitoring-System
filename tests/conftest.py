import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(Path(__file__).parent)]
os.environ.setdefault("NEUROTRACK_DB_PATH", ":memory:")

import pytest  # noqa: E402


@pytest.fixture
def repo():
    from neurotrack.db.repository import Repository
    return Repository(":memory:")


@pytest.fixture
def client(repo):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from neurotrack.config import Settings
    from neurotrack.main import create_app
    return TestClient(create_app(Settings(api_keys=()), repo))


@pytest.fixture
def keyed_client(repo):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from neurotrack.config import Settings
    from neurotrack.main import create_app
    return TestClient(create_app(Settings(api_keys=("test-key-123",)), repo))
