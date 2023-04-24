"""Cache test."""
from typing import Dict, Type, cast

import numpy as np
import pytest
from redis import Redis
from sqlitedict import SqliteDict

from manifest.caches.cache import Cache
from manifest.caches.noop import NoopCache
from manifest.caches.postgres import PostgresCache
from manifest.caches.redis import RedisCache
from manifest.caches.sqlite import SQLiteCache
from manifest.request import DiffusionRequest, LMRequest, Request
from manifest.response import ArrayModelChoice, ModelChoices, Response


def _get_postgres_cache(
    request_type: Type[Request] = LMRequest, cache_args: Dict = {}
) -> Cache:  # type: ignore
    """Get postgres cache."""
    cache_args.update({"cache_user": "", "cache_password": "", "cache_db": ""})
    return PostgresCache(
        "postgres",
        request_type=request_type,
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
    sqlite_cache: str,
    redis_cache: str,
    postgres_cache: str,
    cache_type: str,
    model_choice: ModelChoices,
    model_choice_single: ModelChoices,
    model_choice_arr_int: ModelChoices,
    request_lm: LMRequest,
    request_lm_single: LMRequest,
    request_diff: DiffusionRequest,
) -> None:
    """Test cache save prompt."""
    if cache_type == "sqlite":
        cache = cast(Cache, SQLiteCache(sqlite_cache))
    elif cache_type == "redis":
        cache = cast(Cache, RedisCache(redis_cache))
    elif cache_type == "postgres":
        cache = cast(Cache, _get_postgres_cache())

    response = Response(
        response=model_choice_single,
        cached=False,
        request=request_lm_single,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )

    cache_response = cache.get(request_lm_single.dict())
    assert cache_response is None

    cache.set(request_lm_single.dict(), response.to_dict(drop_request=True))
    cache_response = cache.get(request_lm_single.dict())
    assert cache_response.get_response() == "helloo"
    assert cache_response.is_cached()
    assert cache_response.get_request_obj() == request_lm_single

    response = Response(
        response=model_choice,
        cached=False,
        request=request_lm,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )

    cache_response = cache.get(request_lm.dict())
    assert cache_response is None

    cache.set(request_lm.dict(), response.to_dict(drop_request=True))
    cache_response = cache.get(request_lm.dict())
    assert cache_response.get_response() == ["hello", "bye"]
    assert cache_response.is_cached()
    assert cache_response.get_request_obj() == request_lm

    # Test array
    response = Response(
        response=model_choice_arr_int,
        cached=False,
        request=request_diff,
        usages=None,
        request_type=DiffusionRequest,
        response_type="array",
    )

    if cache_type == "sqlite":
        cache = SQLiteCache(sqlite_cache, request_type=DiffusionRequest)
    elif cache_type == "redis":
        cache = RedisCache(redis_cache, request_type=DiffusionRequest)
    elif cache_type == "postgres":
        cache = _get_postgres_cache(request_type=DiffusionRequest)

    cache_response = cache.get(request_diff.dict())
    assert cache_response is None

    cache.set(request_diff.dict(), response.to_dict(drop_request=True))
    cached_response = cache.get(request_diff.dict())
    assert np.allclose(
        cached_response.get_response()[0],
        cast(ArrayModelChoice, model_choice_arr_int.choices[0]).array,
    )
    assert np.allclose(
        cached_response.get_response()[1],
        cast(ArrayModelChoice, model_choice_arr_int.choices[1]).array,
    )
    assert cached_response.is_cached()
    assert cached_response.get_request_obj() == request_diff

    # Test array byte string
    # Make sure to not hit the cache
    new_request_diff = DiffusionRequest(**request_diff.dict())
    new_request_diff.prompt = ["blahhh", "yayayay"]
    response = Response(
        response=model_choice_arr_int,
        cached=False,
        request=new_request_diff,
        usages=None,
        request_type=DiffusionRequest,
        response_type="array",
    )

    if cache_type == "sqlite":
        cache = SQLiteCache(
            sqlite_cache,
            request_type=DiffusionRequest,
            cache_args={"array_serializer": "byte_string"},
        )
    elif cache_type == "redis":
        cache = RedisCache(
            redis_cache,
            request_type=DiffusionRequest,
            cache_args={"array_serializer": "byte_string"},
        )
    elif cache_type == "postgres":
        cache = _get_postgres_cache(
            request_type=DiffusionRequest,
            cache_args={"array_serializer": "byte_string"},
        )

    cached_response = cache.get(new_request_diff.dict())
    assert cached_response is None

    cache.set(new_request_diff.dict(), response.to_dict(drop_request=True))
    cached_response = cache.get(new_request_diff.dict())
    assert np.allclose(
        cached_response.get_response()[0],
        cast(ArrayModelChoice, model_choice_arr_int.choices[0]).array,
    )
    assert np.allclose(
        cached_response.get_response()[1],
        cast(ArrayModelChoice, model_choice_arr_int.choices[1]).array,
    )
    assert cached_response.is_cached()
    assert cached_response.get_request_obj() == new_request_diff


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
