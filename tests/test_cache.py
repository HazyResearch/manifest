"""Cache test."""
import numpy as np
import pytest
from redis import Redis
from sqlitedict import SqliteDict

from manifest.caches.noop import NoopCache
from manifest.caches.redis import RedisCache
from manifest.caches.sqlite import SQLiteCache


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("redis_cache")
@pytest.mark.parametrize("cache_type", ["sqlite", "redis"])
def test_init(sqlite_cache, redis_cache, cache_type):
    """Test cache initialization."""
    if cache_type == "sqlite":
        cache = SQLiteCache(sqlite_cache)
        assert isinstance(cache.cache, SqliteDict)
    else:
        cache = RedisCache(redis_cache)
        assert isinstance(cache.redis, Redis)


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("redis_cache")
@pytest.mark.parametrize("cache_type", ["sqlite", "redis"])
def test_key_get_and_set(sqlite_cache, redis_cache, cache_type):
    """Test cache key get and set."""
    if cache_type == "sqlite":
        cache = SQLiteCache(sqlite_cache)
    else:
        cache = RedisCache(redis_cache)

    cache.set_key("test", "valueA")
    cache.set_key("testA", "valueB")
    assert cache.get_key("test") == "valueA"
    assert cache.get_key("testA") == "valueB"

    cache.set_key("testA", "valueC")
    assert cache.get_key("testA") == "valueC"

    cache.get_key("test", table="prompt") is None
    cache.set_key("test", "valueA", table="prompt")
    cache.get_key("test", table="prompt") == "valueA"


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("redis_cache")
@pytest.mark.parametrize("cache_type", ["sqlite", "redis"])
def test_get(sqlite_cache, redis_cache, cache_type):
    """Test cache save prompt."""
    if cache_type == "sqlite":
        cache = SQLiteCache(sqlite_cache)
    else:
        cache = RedisCache(redis_cache)
    test_request = {"test": "hello", "testA": "world"}
    compute = lambda: {"choices": [{"text": "hello"}]}

    # response = cache.get(test_request, overwrite_cache=False, compute=compute)
    # assert response.get_response() == "hello"
    # assert not response.is_cached()
    # assert response.get_request() == test_request

    # response = cache.get(test_request, overwrite_cache=False, compute=compute)
    # assert response.get_response() == "hello"
    # assert response.is_cached()
    # assert response.get_request() == test_request

    # response = cache.get(test_request, overwrite_cache=True, compute=compute)
    # assert response.get_response() == "hello"
    # assert not response.is_cached()
    # assert response.get_request() == test_request

    arr = np.random.rand(4, 4)
    test_request = {"test": "hello", "testA": "world of images"}
    compute = lambda: {"choices": [{"array": arr}]}

    # Test array
    if cache_type == "sqlite":
        cache = SQLiteCache(sqlite_cache, client_name="diffuser")
    else:
        cache = RedisCache(redis_cache, client_name="diffuser")
    response = cache.get(test_request, overwrite_cache=False, compute=compute)
    assert np.allclose(response.get_response(), arr)
    assert not response.is_cached()
    assert response.get_request() == test_request


def test_noop_cache():
    """Test cache that is a no-op cache."""
    cache = NoopCache(None)
    cache.set_key("test", "valueA")
    cache.set_key("testA", "valueB")
    assert cache.get_key("test") is None
    assert cache.get_key("testA") is None

    cache.set_key("testA", "valueC")
    assert cache.get_key("testA") is None

    cache.get_key("test", table="prompt") is None
    cache.set_key("test", "valueA", table="prompt")
    cache.get_key("test", table="prompt") is None

    # Assert always not cached
    test_request = {"test": "hello", "testA": "world"}
    compute = lambda: {"choices": [{"text": "hello"}]}

    response = cache.get(test_request, overwrite_cache=False, compute=compute)
    assert response.get_response() == "hello"
    assert not response.is_cached()
    assert response.get_request() == test_request

    response = cache.get(test_request, overwrite_cache=False, compute=compute)
    assert response.get_response() == "hello"
    assert not response.is_cached()
    assert response.get_request() == test_request
