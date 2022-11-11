"""Manifest test."""
import pytest

from manifest import Manifest, Response
from manifest.caches.cache import request_to_key
from manifest.caches.noop import NoopCache
from manifest.caches.sqlite import SQLiteCache
from manifest.clients.dummy import DummyClient
from manifest.session import Session


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("session_cache")
def test_init(sqlite_cache, session_cache):
    """Test manifest initialization."""
    with pytest.raises(ValueError) as exc_info:
        Manifest(
            client_name="dummy",
            cache_name="sqlite",
            cache_connection=sqlite_cache,
            sep_tok="",
        )
    assert str(exc_info.value) == "[('sep_tok', '')] arguments are not recognized."

    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )
    assert manifest.client_name == "dummy"
    assert isinstance(manifest.client, DummyClient)
    assert isinstance(manifest.cache, SQLiteCache)
    assert manifest.session is None
    assert manifest.client.n == 1
    assert manifest.stop_token == ""

    manifest = Manifest(
        client_name="dummy",
        cache_name="noop",
        n=3,
        stop_token="\n",
        session_id="_default",
    )
    assert manifest.client_name == "dummy"
    assert isinstance(manifest.client, DummyClient)
    assert isinstance(manifest.cache, NoopCache)
    assert isinstance(manifest.session, Session)
    assert manifest.client.n == 3
    assert manifest.stop_token == "\n"


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("session_cache")
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("return_response", [True, False])
def test_run(sqlite_cache, session_cache, n, return_response):
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        n=n,
    )

    prompt = "This is a prompt"
    with pytest.raises(ValueError) as exc_info:
        result = manifest.run(prompt, return_response=return_response, bad_input=5)
    assert str(exc_info.value) == "[('bad_input', 5)] arguments are not recognized."

    # Allow params in the request object but not in the client to go through
    assert "top_k" not in manifest.client.PARAMS
    result = manifest.run(prompt, return_response=return_response, top_k=5)
    assert result is not None

    prompt = "This is a prompt"
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
                    "engine": "dummy",
                    "num_results": n,
                }
            )
        )
        is not None
    )
    if n == 1:
        assert res == "hello"
    else:
        assert res == ["hello", "hello"]

    prompt = "This is a prompt"
    result = manifest.run(prompt, run_id="34", return_response=return_response)
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
                    "engine": "dummy",
                    "num_results": n,
                    "run_id": "34",
                }
            )
        )
        is not None
    )
    if n == 1:
        assert res == "hello"
    else:
        assert res == ["hello", "hello"]

    prompt = "Hello is a prompt"
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
                    "prompt": "Hello is a prompt",
                    "engine": "dummy",
                    "num_results": n,
                }
            )
        )
        is not None
    )
    if n == 1:
        assert res == "hello"
    else:
        assert res == ["hello", "hello"]

    prompt = "Hello is a prompt"
    result = manifest.run(prompt, stop_token="ll", return_response=return_response)
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
                    "engine": "dummy",
                    "num_results": n,
                }
            )
        )
        is not None
    )
    if n == 1:
        assert res == "he"
    else:
        assert res == ["he", "he"]


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("session_cache")
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("return_response", [True, False])
def test_batch_run(sqlite_cache, session_cache, n, return_response):
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        n=n,
    )
    prompt = ["This is a prompt"]
    if n == 2:
        with pytest.raises(ValueError) as exc_info:
            result = manifest.run(prompt, return_response=return_response)
        assert str(exc_info.value) == "Batch mode does not support n > 1."
    else:
        result = manifest.run(prompt, return_response=return_response)
        if return_response:
            res = result.get_response(manifest.stop_token, is_batch=True)
        else:
            res = result
        assert res == ["hello"]

        prompt = ["Hello is a prompt", "Hello is a prompt"]
        result = manifest.run(prompt, return_response=return_response)
        if return_response:
            res = result.get_response(manifest.stop_token, is_batch=True)
        else:
            res = result
        assert res == ["hello", "hello"]

        prompt = ["Hello is a prompt", "Hello is a prompt"]
        result = manifest.run(prompt, stop_token="ll", return_response=return_response)
        if return_response:
            res = result.get_response(stop_token="ll", is_batch=True)
        else:
            res = result
        assert res == ["he", "he"]


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.usefixtures("session_cache")
@pytest.mark.parametrize("return_response", [True, False])
def test_choices_run(sqlite_cache, session_cache, return_response):
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )

    prompt = "This is a prompt"
    # Dummy client will always return first choice
    choices = ["cat", "dog"]
    result = manifest.run(prompt, gold_choices=choices, return_response=return_response)
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
                    "gold_choices": ["cat", "dog"],
                    "engine": "dummy",
                }
            )
        )
        is not None
    )
    assert res == "cat"

    prompt = "Hello is a prompt"
    choices = ["cat", "dog"]
    result = manifest.run(prompt, gold_choices=choices, return_response=return_response)
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
                    "gold_choices": ["cat", "dog"],
                    "engine": "dummy",
                }
            )
        )
        is not None
    )
    assert res == "cat"

    prompt = "Hello is a prompt"
    choices = ["callt", "dog"]
    result = manifest.run(
        prompt,
        gold_choices=choices,
        stop_token="ll",
        return_response=return_response,
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
                    "gold_choices": ["cat", "dog"],
                    "engine": "dummy",
                }
            )
        )
        is not None
    )
    assert res == "ca"

    prompt = ["Hello is a prompt", "Hello is a prompt"]
    choices = ["callt", "dog"]
    result = manifest.run(
        prompt,
        gold_choices=choices,
        stop_token="ll",
        return_response=return_response,
    )
    if return_response:
        assert isinstance(result, Response)
        res = result.get_response(stop_token="ll", is_batch=True)
    else:
        res = result
    assert (
        manifest.cache.get_key(
            request_to_key(
                {
                    "prompt": ["Hello is a prompt", "Hello is a prompt"],
                    "gold_choices": ["callt", "dog"],
                    "engine": "dummy",
                }
            )
        )
        is not None
    )
    assert res == ["ca", "ca"]


@pytest.mark.usefixtures("session_cache")
def test_log_query(session_cache):
    """Test manifest session logging."""
    manifest = Manifest(client_name="dummy", cache_name="noop", session_id="_default")
    prompt = "This is a prompt"
    _ = manifest.run(prompt, return_response=False)
    query_key = {
        "prompt": "This is a prompt",
        "engine": "dummy",
        "num_results": 1,
    }
    response_key = {
        "cached": False,
        "request_params": query_key,
        "response": {"choices": [{"text": "hello"}]},
    }
    assert manifest.get_last_queries(1) == [("This is a prompt", "hello")]
    assert manifest.get_last_queries(1, return_raw_values=True) == [
        (query_key, response_key)
    ]
    assert manifest.get_last_queries(3, return_raw_values=True) == [
        (query_key, response_key)
    ]
    prior_cache_item = (query_key, response_key)

    prompt = ["This is a prompt", "This is a prompt2"]
    _ = manifest.run(prompt, return_response=False)
    query_key = {
        "prompt": ["This is a prompt", "This is a prompt2"],
        "engine": "dummy",
        "num_results": 1,
    }
    response_key = {
        "cached": False,
        "request_params": query_key,
        "response": {"choices": [{"text": "hello"}, {"text": "hello"}]},
    }
    assert manifest.get_last_queries(1) == [
        (["This is a prompt", "This is a prompt2"], ["hello", "hello"])
    ]
    assert manifest.get_last_queries(1, return_raw_values=True) == [
        (query_key, response_key)
    ]
    assert manifest.get_last_queries(3, return_raw_values=True) == [
        prior_cache_item,
        (query_key, response_key),
    ]

    # Test no session
    manifest = Manifest(
        client_name="dummy",
        cache_name="noop",
    )
    prompt = "This is a prompt"
    _ = manifest.run(prompt, return_response=False)
    with pytest.raises(ValueError) as exc_info:
        manifest.get_last_queries(1)
    assert (
        str(exc_info.value)
        == "Session was not initialized. Set `session_id` when loading Manifest."
    )
