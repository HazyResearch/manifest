"""Manifest test."""
import pytest

from manifest import Manifest, Prompt
from manifest.caches.cache import request_to_key
from manifest.caches.sqlite import SQLiteCache
from manifest.clients.dummy import DummyClient


@pytest.mark.usefixtures("sqlite_cache")
def test_init(sqlite_cache):
    """Test manifest initialization."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )
    assert manifest.client_name == "dummy"
    assert isinstance(manifest.client, DummyClient)
    assert isinstance(manifest.cache, SQLiteCache)

    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        num_results=3,
    )
    assert manifest.client_name == "dummy"
    assert isinstance(manifest.client, DummyClient)
    assert isinstance(manifest.cache, SQLiteCache)
    assert manifest.client.num_results == 3


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.parametrize("num_results", [1, 2])
def test_run(sqlite_cache, num_results):
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        num_results=num_results,
    )
    prompt = Prompt("This is a prompt")
    res = manifest.run(prompt)
    assert (
        manifest.cache.get_key(
            request_to_key(
                {
                    "prompt": "This is a prompt",
                    "client_name": "dummy",
                    "num_results": num_results,
                }
            )
        )
        is not None
    )
    if num_results == 1:
        assert res == "hello"
    else:
        assert res == ["hello", "hello"]

    prompt = Prompt(lambda x: f"{x} is a prompt")
    res = manifest.run(prompt, "Hello")
    assert (
        manifest.cache.get_key(
            request_to_key(
                {
                    "prompt": "Hello is a prompt",
                    "client_name": "dummy",
                    "num_results": num_results,
                }
            )
        )
        is not None
    )
    if num_results == 1:
        assert res == "hello"
    else:
        assert res == ["hello", "hello"]


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.parametrize("num_results", [1, 2])
def test_batch_run(sqlite_cache, num_results):
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        num_results=num_results,
    )
    prompt = Prompt("This is a prompt")
    res = manifest.run_batch(prompt)
    if num_results == 1:
        assert res == ["hello"]
    else:
        assert res == [["hello", "hello"]]

    prompt = Prompt(lambda x: f"{x} is a prompt")
    res = manifest.run_batch(prompt, ["Hello", "Hello"])
    if num_results == 1:
        assert res == ["hello", "hello"]
    else:
        assert res == [["hello", "hello"], ["hello", "hello"]]
