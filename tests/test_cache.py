"""Cache test."""
from typing import Dict, cast

import numpy as np
import pytest
from redis import Redis
from sqlitedict import SqliteDict

from manifest.caches.cache import Cache
from manifest.caches.noop import NoopCache
from manifest.caches.postgres import PostgresCache
from manifest.caches.redis import RedisCache
from manifest.caches.sqlite import SQLiteCache


def _get_postgres_cache(
    client_name: str = "", cache_args: Dict = {}
) -> Cache:  # type: ignore
    """Get postgres cache."""
    cache_args.update({"cache_user": "", "cache_password": "", "cache_db": ""})
    return PostgresCache(
        "postgres",
        client_name=client_name,
        cache_args=cache_args,
    )


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("redis_cache")
@pytest.mark.usefixtures("postgres_cache")
@pytest.mark.parametrize("cache_type", ["sqlite", "redis", "postgres"])
def test_init(
    sqlite_cache: str, redis_cache: str, postgres_cache: str, cache_type: str
) -> None:
    """Test cache initialization."""
    if cache_type == "sqlite":
        sql_cache_obj = SQLiteCache(sqlite_cache)
        assert isinstance(sql_cache_obj.cache, SqliteDict)
    elif cache_type == "redis":
        redis_cache_obj = RedisCache(redis_cache)
        assert isinstance(redis_cache_obj.redis, Redis)
    elif cache_type == "postgres":
        postgres_cache_obj = _get_postgres_cache()
        isinstance(postgres_cache_obj, PostgresCache)


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("redis_cache")
@pytest.mark.usefixtures("postgres_cache")
@pytest.mark.parametrize("cache_type", ["sqlite", "postgres", "redis"])
def test_key_get_and_set(
    sqlite_cache: str, redis_cache: str, postgres_cache: str, cache_type: str
) -> None:
    """Test cache key get and set."""
    if cache_type == "sqlite":
        cache = cast(Cache, SQLiteCache(sqlite_cache))
    elif cache_type == "redis":
        cache = cast(Cache, RedisCache(redis_cache))
    elif cache_type == "postgres":
        cache = cast(Cache, _get_postgres_cache())

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
@pytest.mark.usefixtures("postgres_cache")
@pytest.mark.parametrize("cache_type", ["sqlite", "redis", "postgres"])
def test_get(
    sqlite_cache: str, redis_cache: str, postgres_cache: str, cache_type: str
) -> None:
    """Test cache save prompt."""
    if cache_type == "sqlite":
        cache = cast(Cache, SQLiteCache(sqlite_cache))
    elif cache_type == "redis":
        cache = cast(Cache, RedisCache(redis_cache))
    elif cache_type == "postgres":
        cache = cast(Cache, _get_postgres_cache())

    test_request = {"test": "hello", "testA": "world"}
    test_response = {"choices": [{"text": "hello"}]}

    response = cache.get(test_request)
    assert response is None

    cache.set(test_request, test_response)
    response = cache.get(test_request)
    assert response.get_response() == "hello"
    assert response.is_cached()
    assert response.get_request() == test_request

    # Test array
    arr = np.random.rand(4, 4)
    test_request = {"test": "hello", "testA": "world of images"}
    compute_arr_response = {"choices": [{"array": arr}]}

    if cache_type == "sqlite":
        cache = SQLiteCache(sqlite_cache, client_name="diffuser")
    elif cache_type == "redis":
        cache = RedisCache(redis_cache, client_name="diffuser")
    elif cache_type == "postgres":
        cache = _get_postgres_cache(client_name="diffuser")

    response = cache.get(test_request)
    assert response is None

    cache.set(test_request, compute_arr_response)
    response = cache.get(test_request)
    assert np.allclose(response.get_response(), arr)
    assert response.is_cached()
    assert response.get_request() == test_request

    # Test array byte string
    arr = np.random.rand(4, 4)
    test_request = {"test": "hello", "testA": "world of images 2"}
    compute_arr_response = {"choices": [{"array": arr}]}

    if cache_type == "sqlite":
        cache = SQLiteCache(
            sqlite_cache,
            client_name="diffuser",
            cache_args={"array_serializer": "byte_string"},
        )
    elif cache_type == "redis":
        cache = RedisCache(
            redis_cache,
            client_name="diffuser",
            cache_args={"array_serializer": "byte_string"},
        )
    elif cache_type == "postgres":
        cache = _get_postgres_cache(
            client_name="diffuser", cache_args={"array_serializer": "byte_string"}
        )

    response = cache.get(test_request)
    assert response is None

    cache.set(test_request, compute_arr_response)
    response = cache.get(test_request)
    assert np.allclose(response.get_response(), arr)
    assert response.is_cached()
    assert response.get_request() == test_request


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("redis_cache")
@pytest.mark.usefixtures("postgres_cache")
@pytest.mark.parametrize("cache_type", ["sqlite", "redis", "postgres"])
def test_get_batch_prompt(
    sqlite_cache: str, redis_cache: str, postgres_cache: str, cache_type: str
) -> None:
    """Test cache save prompt."""
    if cache_type == "sqlite":
        cache = cast(Cache, SQLiteCache(sqlite_cache))
    elif cache_type == "redis":
        cache = cast(Cache, RedisCache(redis_cache))
    elif cache_type == "postgres":
        cache = cast(Cache, _get_postgres_cache())

    test_request = {"test": ["hello", "goodbye"], "testA": "world"}
    test_response = {"choices": [{"text": "hello"}, {"text": "goodbye"}]}

    response = cache.get(test_request)
    assert response is None

    cache.set(test_request, test_response)
    response = cache.get(test_request)
    assert response.get_response() == ["hello", "goodbye"]
    assert response.is_cached()
    assert response.get_request() == test_request

    # Test arrays
    arr = np.random.rand(4, 4)
    arr2 = np.random.rand(4, 4)
    test_request = {"test": ["hello", "goodbye"], "testA": "world of images"}
    compute_arr_response = {"choices": [{"array": arr}, {"array": arr2}]}

    if cache_type == "sqlite":
        cache = SQLiteCache(sqlite_cache, client_name="diffuser")
    elif cache_type == "redis":
        cache = RedisCache(redis_cache, client_name="diffuser")
    elif cache_type == "postgres":
        cache = _get_postgres_cache(client_name="diffuser")

    response = cache.get(test_request)
    assert response is None

    cache.set(test_request, compute_arr_response)
    response = cache.get(test_request)
    assert np.allclose(response.get_response()[0], arr)
    assert np.allclose(response.get_response()[1], arr2)
    assert response.is_cached()
    assert response.get_request() == test_request

    # Test arrays byte serializer
    arr = np.random.rand(4, 4)
    arr2 = np.random.rand(4, 4)
    test_request = {"test": ["hello", "goodbye"], "testA": "world of images 2"}
    compute_arr_response = {"choices": [{"array": arr}, {"array": arr2}]}

    if cache_type == "sqlite":
        cache = SQLiteCache(
            sqlite_cache,
            client_name="diffuser",
            cache_args={"array_serializer": "byte_string"},
        )
    elif cache_type == "redis":
        cache = RedisCache(
            redis_cache,
            client_name="diffuser",
            cache_args={"array_serializer": "byte_string"},
        )
    elif cache_type == "postgres":
        cache = _get_postgres_cache(
            client_name="diffuser", cache_args={"array_serializer": "byte_string"}
        )

    response = cache.get(test_request)
    assert response is None

    cache.set(test_request, compute_arr_response)
    response = cache.get(test_request)
    assert np.allclose(response.get_response()[0], arr)
    assert np.allclose(response.get_response()[1], arr2)
    assert response.is_cached()
    assert response.get_request() == test_request


def test_noop_cache() -> None:
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
    test_response = {"choices": [{"text": "hello"}]}

    response = cache.get(test_request)
    assert response is None

    cache.set(test_request, test_response)
    response = cache.get(test_request)
    assert response is None
