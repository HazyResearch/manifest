"""Manifest test."""
import pytest

from manifest import Manifest, Prompt, Response
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
    assert manifest.client.num_results == 1
    assert manifest.stop_token == ""

    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        num_results=3,
        stop_token="\n",
    )
    assert manifest.client_name == "dummy"
    assert isinstance(manifest.client, DummyClient)
    assert isinstance(manifest.cache, SQLiteCache)
    assert manifest.client.num_results == 3
    assert manifest.stop_token == "\n"


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.parametrize("num_results", [1, 2])
@pytest.mark.parametrize("return_response", [True, False])
def test_run(sqlite_cache, num_results, return_response):
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        num_results=num_results,
    )
    prompt = Prompt("This is a prompt")
    result = manifest.run(prompt, return_response=return_response)
    if return_response:
        assert isinstance(result, Response)
        res = result.get_response(manifest.stop_token)
    else:
        res = result
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
    result = manifest.run(prompt, "Hello", return_response=return_response)
    if return_response:
        assert isinstance(result, Response)
        res = result.get_response(manifest.stop_token)
    else:
        res = result
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

    prompt = Prompt(lambda x: f"{x} is a prompt")
    result = manifest.run(
        prompt, "Hello", stop_token="ll", return_response=return_response
    )
    if return_response:
        assert isinstance(result, Response)
        res = result.get_response(stop_token="ll")
    else:
        res = result
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
        assert res == "he"
    else:
        assert res == ["he", "he"]


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.parametrize("num_results", [1, 2])
@pytest.mark.parametrize("return_response", [True, False])
def test_batch_run(sqlite_cache, num_results, return_response):
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        num_results=num_results,
    )
    prompt = Prompt("This is a prompt")
    result = manifest.run_batch(prompt, return_response=return_response)
    if return_response:
        res = [r.get_response(manifest.stop_token) for r in result]
    else:
        res = result
    if num_results == 1:
        assert res == ["hello"]
    else:
        assert res == [["hello", "hello"]]

    prompt = Prompt(lambda x: f"{x} is a prompt")
    result = manifest.run_batch(
        prompt, ["Hello", "Hello"], return_response=return_response
    )
    if return_response:
        res = [r.get_response(manifest.stop_token) for r in result]
    else:
        res = result
    if num_results == 1:
        assert res == ["hello", "hello"]
    else:
        assert res == [["hello", "hello"], ["hello", "hello"]]

    prompt = Prompt(lambda x: f"{x} is a prompt")
    result = manifest.run_batch(
        prompt, ["Hello", "Hello"], stop_token="ll", return_response=return_response
    )
    if return_response:
        res = [r.get_response(stop_token="ll") for r in result]
    else:
        res = result
    if num_results == 1:
        assert res == ["he", "he"]
    else:
        assert res == [["he", "he"], ["he", "he"]]
