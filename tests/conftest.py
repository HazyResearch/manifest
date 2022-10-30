"""Setup for all tests."""
import os
import shutil
from pathlib import Path

import pytest
import redis


@pytest.fixture
def sqlite_cache(tmp_path):
    """Sqlite Cache."""
    cache = str(tmp_path / "sqlite_cache.sqlite")
    yield cache
    shutil.rmtree(cache, ignore_errors=True)


@pytest.fixture
def redis_cache():
    """Redis cache."""
    host = os.environ.get("REDIS_HOST", "localhost")
    port = os.environ.get("REDIS_PORT", 6379)
    yield f"{host}:{port}"
    # Clear out the database
    try:
        db = redis.Redis(host=host, port=port)
        db.flushdb()
    # For better local testing, pass if redis DB not started
    except OSError:
        pass


@pytest.fixture
def session_cache(tmpdir):
    """Session cache dir."""
    os.environ["MANIFEST_HOME"] = str(tmpdir)
    yield Path(tmpdir)
