"""Cache test."""
import pytest
from sqlitedict import SqliteDict

from manifest.caches.redis import RedisCache
from manifest.caches.sqlite import SQLiteCache
from manifest.clients import Response


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("redis_cache")
@pytest.mark.parametrize("cache_type", ["sqlite"])
def test_init(sqlite_cache, redis_cache, cache_type):
    """Test cache initialization."""
    if cache_type == "sqlite":
        cache = SQLiteCache(sqlite_cache)
        assert isinstance(cache.cache, SqliteDict)
        assert isinstance(cache.prompt_cache, SqliteDict)
    else:
        cache = RedisCache(redis_cache)


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("redis_cache")
@pytest.mark.parametrize("cache_type", ["sqlite"])
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
@pytest.mark.parametrize("cache_type", ["sqlite"])
def test_get(sqlite_cache, redis_cache, cache_type):
    """Test cache save prompt."""
    if cache_type == "sqlite":
        cache = SQLiteCache(sqlite_cache)
    else:
        cache = RedisCache(redis_cache)
    test_request = {"test": "hello", "testA": "world"}
    compute = lambda: Response({"choices": [{"text": "hello"}]})

    response, cached = cache.get(test_request, overwrite_cache=False, compute=compute)
    assert response.get_results() == "hello"
    assert not cached

    response, cached = cache.get(test_request, overwrite_cache=False, compute=compute)
    assert response.get_results() == "hello"
    assert cached

    response, cached = cache.get(test_request, overwrite_cache=True, compute=compute)
    assert response.get_results() == "hello"
    assert not cached
