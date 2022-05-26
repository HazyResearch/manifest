"""Setup for all tests."""
import os
import shutil

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
    if "CI" not in os.environ:
        # Give a clear warning on setting REDIS_PORT before running tests.
        try:
            port = os.environ["REDIS_PORT"]
        except KeyError:
            raise KeyError(
                "Set REDIS_PORT env var to the instance you want to use "
                + "for testing. Note that doing so WILL delete the db at "
                + "localhost:REDIS_PORT, db=0, so BE CAREFUL."
            )
        host = os.environ.get("REDIS_HOST", "localhost")
    else:
        host = os.environ.get("REDIS_HOST", "localhost")
        port = os.environ.get("REDIS_PORT", 6379)
    yield f"{host}:{port}"
    # Clear out the database
    db = redis.Redis(host=host, port=port)
    db.flushdb()
