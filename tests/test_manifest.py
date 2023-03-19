"""Manifest test."""
import asyncio
import os
from typing import cast

import pytest
import requests

from manifest import Manifest, Response
from manifest.caches.noop import NoopCache
from manifest.caches.sqlite import SQLiteCache
from manifest.clients.dummy import DummyClient

URL = "http://localhost:6000"
try:
    _ = requests.post(URL + "/params").json()
    MODEL_ALIVE = True
except Exception:
    MODEL_ALIVE = False

OPENAI_ALIVE = os.environ.get("OPENAI_API_KEY") is not None


@pytest.mark.usefixtures("sqlite_cache")
def test_init(sqlite_cache: str) -> None:
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
    assert manifest.client.n == 1  # type: ignore
    assert manifest.stop_token == ""

    manifest = Manifest(
        client_name="dummy",
        cache_name="noop",
        n=3,
        stop_token="\n",
    )
    assert manifest.client_name == "dummy"
    assert isinstance(manifest.client, DummyClient)
    assert isinstance(manifest.cache, NoopCache)
    assert manifest.client.n == 3  # type: ignore
    assert manifest.stop_token == "\n"


@pytest.mark.usefixtures("sqlite_cache")
def test_change_manifest(sqlite_cache: str) -> None:
    """Test manifest change."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )

    manifest.change_client()
    assert manifest.client_name == "dummy"
    assert isinstance(manifest.client, DummyClient)
    assert isinstance(manifest.cache, SQLiteCache)
    assert manifest.client.n == 1  # type: ignore
    assert manifest.stop_token == ""

    manifest.change_client(stop_token="\n")
    assert manifest.client_name == "dummy"
    assert isinstance(manifest.client, DummyClient)
    assert isinstance(manifest.cache, SQLiteCache)
    assert manifest.client.n == 1  # type: ignore
    assert manifest.stop_token == "\n"


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("return_response", [True, False])
def test_run(sqlite_cache: str, n: int, return_response: bool) -> None:
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
        result = cast(Response, result)
        assert len(result.get_json_response()["usage"]) == len(
            result.get_json_response()["choices"]
        )
        res = result.get_response(manifest.stop_token)
    else:
        res = cast(str, result)
    assert (
        manifest.cache.get(
            {
                "prompt": "This is a prompt",
                "engine": "dummy",
                "num_results": n,
            },
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
        result = cast(Response, result)
        assert len(result.get_json_response()["usage"]) == len(
            result.get_json_response()["choices"]
        )
        res = result.get_response(manifest.stop_token)
    else:
        res = cast(str, result)
    assert (
        manifest.cache.get(
            {
                "prompt": "This is a prompt",
                "engine": "dummy",
                "num_results": n,
                "run_id": "34",
            }
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
        result = cast(Response, result)
        assert len(result.get_json_response()["usage"]) == len(
            result.get_json_response()["choices"]
        )
        res = result.get_response(manifest.stop_token)
    else:
        res = cast(str, result)
    assert (
        manifest.cache.get(
            {
                "prompt": "Hello is a prompt",
                "engine": "dummy",
                "num_results": n,
            },
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
        result = cast(Response, result)
        assert len(result.get_json_response()["usage"]) == len(
            result.get_json_response()["choices"]
        )
        res = result.get_response(stop_token="ll")
    else:
        res = cast(str, result)
    assert (
        manifest.cache.get(
            {
                "prompt": "Hello is a prompt",
                "engine": "dummy",
                "num_results": n,
            },
        )
        is not None
    )
    if n == 1:
        assert res == "he"
    else:
        assert res == ["he", "he"]


@pytest.mark.usefixtures("sqlite_cache")
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("return_response", [True, False])
def test_batch_run(sqlite_cache: str, n: int, return_response: bool) -> None:
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
            assert isinstance(result, Response)
            result = cast(Response, result)
            assert len(result.get_json_response()["usage"]) == len(
                result.get_json_response()["choices"]
            )
            res = result.get_response(manifest.stop_token, is_batch=True)
        else:
            res = cast(str, result)
        assert res == ["hello"]
        assert (
            manifest.cache.get(
                {
                    "prompt": "This is a prompt",
                    "engine": "dummy",
                    "num_results": n,
                },
            )
            is not None
        )

        prompt = ["Hello is a prompt", "Hello is a prompt"]
        result = manifest.run(prompt, return_response=return_response)
        if return_response:
            assert isinstance(result, Response)
            result = cast(Response, result)
            assert len(result.get_json_response()["usage"]) == len(
                result.get_json_response()["choices"]
            )
            res = result.get_response(manifest.stop_token, is_batch=True)
        else:
            res = cast(str, result)
        assert res == ["hello", "hello"]
        assert (
            manifest.cache.get(
                {
                    "prompt": "Hello is a prompt",
                    "engine": "dummy",
                    "num_results": n,
                },
            )
            is not None
        )

        result = manifest.run(prompt, return_response=True)
        res = cast(Response, result).get_response(manifest.stop_token, is_batch=True)
        assert cast(Response, result).is_cached()

        assert (
            manifest.cache.get(
                {
                    "prompt": "New prompt",
                    "engine": "dummy",
                    "num_results": n,
                },
            )
            is None
        )
        prompt = ["This is a prompt", "New prompt"]
        result = manifest.run(prompt, return_response=return_response)
        if return_response:
            assert isinstance(result, Response)
            result = cast(Response, result)
            assert len(result.get_json_response()["usage"]) == len(
                result.get_json_response()["choices"]
            )
            res = result.get_response(manifest.stop_token, is_batch=True)
            # Cached because one item is in cache
            assert result.is_cached()
        else:
            res = cast(str, result)
        assert res == ["hello", "hello"]

        prompt = ["Hello is a prompt", "Hello is a prompt"]
        result = manifest.run(prompt, stop_token="ll", return_response=return_response)
        if return_response:
            assert isinstance(result, Response)
            result = cast(Response, result)
            assert len(result.get_json_response()["usage"]) == len(
                result.get_json_response()["choices"]
            )
            res = result.get_response(stop_token="ll", is_batch=True)
        else:
            res = cast(str, result)
        assert res == ["he", "he"]


@pytest.mark.usefixtures("sqlite_cache")
def test_abatch_run(sqlite_cache: str) -> None:
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )
    prompt = ["This is a prompt"]
    result = cast(
        Response, asyncio.run(manifest.arun_batch(prompt, return_response=True))
    )

    assert len(result.get_json_response()["usage"]) == len(
        result.get_json_response()["choices"]
    )
    res = result.get_response(manifest.stop_token, is_batch=True)
    assert res == ["hello"]
    assert (
        manifest.cache.get(
            {
                "prompt": "This is a prompt",
                "engine": "dummy",
                "num_results": 1,
            },
        )
        is not None
    )

    prompt = ["Hello is a prompt", "Hello is a prompt"]
    result = cast(
        Response, asyncio.run(manifest.arun_batch(prompt, return_response=True))
    )

    assert len(result.get_json_response()["usage"]) == len(
        result.get_json_response()["choices"]
    )
    res = result.get_response(manifest.stop_token, is_batch=True)
    assert res == ["hello", "hello"]
    assert (
        manifest.cache.get(
            {
                "prompt": "Hello is a prompt",
                "engine": "dummy",
                "num_results": 1,
            },
        )
        is not None
    )

    result = cast(
        Response, asyncio.run(manifest.arun_batch(prompt, return_response=True))
    )

    assert len(result.get_json_response()["usage"]) == len(
        result.get_json_response()["choices"]
    )
    res = result.get_response(manifest.stop_token, is_batch=True)
    assert result.is_cached()

    assert (
        manifest.cache.get(
            {
                "prompt": "New prompt",
                "engine": "dummy",
                "num_results": 1,
            },
        )
        is None
    )
    prompt = ["This is a prompt", "New prompt"]
    result = cast(
        Response, asyncio.run(manifest.arun_batch(prompt, return_response=True))
    )

    assert len(result.get_json_response()["usage"]) == len(
        result.get_json_response()["choices"]
    )
    res = result.get_response(manifest.stop_token, is_batch=True)
    # Cached because one item is in cache
    assert result.is_cached()
    assert res == ["hello", "hello"]

    prompt = ["Hello is a prompt", "Hello is a prompt"]
    result = cast(
        Response, asyncio.run(manifest.arun_batch(prompt, return_response=True))
    )

    assert len(result.get_json_response()["usage"]) == len(
        result.get_json_response()["choices"]
    )
    res = result.get_response(stop_token="ll", is_batch=True)
    assert res == ["he", "he"]


@pytest.mark.usefixtures("sqlite_cache")
def test_score_run(sqlite_cache: str) -> None:
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )

    prompt = "This is a prompt"
    result = manifest.score_prompt(prompt)
    assert (
        manifest.cache.get(
            {
                "prompt": "This is a prompt",
                "engine": "dummy",
                "num_results": 1,
                "request_type": "score_prompt",
            },
        )
        is not None
    )
    assert result == {
        "generation_key": "choices",
        "logits_key": "token_logprobs",
        "item_key": "text",
        "item_dtype": None,
        "response": {"choices": [{"text": "This is a prompt", "logprob": 0.3}]},
        "cached": False,
        "request_params": {
            "prompt": "This is a prompt",
            "engine": "dummy",
            "num_results": 1,
            "request_type": "score_prompt",
        },
    }

    prompt_list = ["Hello is a prompt", "Hello is another prompt"]
    result = manifest.score_prompt(prompt_list)
    assert (
        manifest.cache.get(
            {
                "prompt": "Hello is a prompt",
                "engine": "dummy",
                "num_results": 1,
                "request_type": "score_prompt",
            },
        )
        is not None
    )
    assert (
        manifest.cache.get(
            {
                "prompt": "Hello is another prompt",
                "engine": "dummy",
                "num_results": 1,
                "request_type": "score_prompt",
            },
        )
        is not None
    )
    assert result == {
        "generation_key": "choices",
        "logits_key": "token_logprobs",
        "item_key": "text",
        "item_dtype": None,
        "response": {
            "choices": [
                {"text": "Hello is a prompt", "logprob": 0.3},
                {"text": "Hello is another prompt", "logprob": 0.3},
            ]
        },
        "cached": False,
        "request_params": {
            "prompt": ["Hello is a prompt", "Hello is another prompt"],
            "engine": "dummy",
            "num_results": 1,
            "request_type": "score_prompt",
        },
    }


@pytest.mark.skipif(not MODEL_ALIVE, reason=f"No model at {URL}")
@pytest.mark.usefixtures("sqlite_cache")
def test_local_huggingface(sqlite_cache: str) -> None:
    """Test local huggingface client."""
    client = Manifest(
        client_name="huggingface",
        client_connection=URL,
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )

    res = client.run("Why are there apples?")
    assert isinstance(res, str) and len(res) > 0

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert isinstance(response.get_response(), str) and len(response.get_response()) > 0
    assert response.is_cached() is True

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert response.is_cached() is True

    res_list = client.run(["Why are there apples?", "Why are there bananas?"])
    assert isinstance(res_list, list) and len(res_list) == 2

    response = cast(
        Response, client.run("Why are there bananas?", return_response=True)
    )
    assert response.is_cached() is True

    res_list = asyncio.run(
        client.arun_batch(["Why are there pears?", "Why are there oranges?"])
    )
    assert isinstance(res_list, list) and len(res_list) == 2

    response = cast(
        Response, client.run("Why are there oranges?", return_response=True)
    )
    assert response.is_cached() is True

    scores = client.score_prompt("Why are there apples?")
    assert isinstance(scores, dict) and len(scores) > 0
    assert scores["cached"] is False
    assert len(scores["response"]["choices"][0]["token_logprobs"]) == len(
        scores["response"]["choices"][0]["tokens"]
    )

    scores = client.score_prompt(["Why are there apples?", "Why are there bananas?"])
    assert isinstance(scores, dict) and len(scores) > 0
    assert scores["cached"] is True
    assert len(scores["response"]["choices"][0]["token_logprobs"]) == len(
        scores["response"]["choices"][0]["tokens"]
    )
    assert len(scores["response"]["choices"][0]["token_logprobs"]) == len(
        scores["response"]["choices"][0]["tokens"]
    )


@pytest.mark.skipif(not OPENAI_ALIVE, reason="No openai key set")
@pytest.mark.usefixtures("sqlite_cache")
def test_openai(sqlite_cache: str) -> None:
    """Test openai client."""
    client = Manifest(
        client_name="openai",
        engine="text-ada-001",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        temperature=0.0,
    )

    res = client.run("Why are there apples?")
    assert isinstance(res, str) and len(res) > 0

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert isinstance(response.get_response(), str) and len(response.get_response()) > 0
    assert response.is_cached() is True
    assert "usage" in response.get_json_response()
    assert response.get_json_response()["usage"][0]["total_tokens"] == 15

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert response.is_cached() is True

    res_list = client.run(["Why are there apples?", "Why are there bananas?"])
    assert isinstance(res_list, list) and len(res_list) == 2

    response = cast(
        Response,
        client.run(
            ["Why are there apples?", "Why are there mangos?"], return_response=True
        ),
    )
    assert (
        isinstance(response.get_response(), list) and len(response.get_response()) == 2
    )
    assert (
        "usage" in response.get_json_response()
        and len(response.get_json_response()["usage"]) == 2
    )
    assert response.get_json_response()["usage"][0]["total_tokens"] == 15
    assert response.get_json_response()["usage"][1]["total_tokens"] == 16

    response = cast(
        Response, client.run("Why are there bananas?", return_response=True)
    )
    assert response.is_cached() is True

    res_list = asyncio.run(
        client.arun_batch(["Why are there pears?", "Why are there oranges?"])
    )
    assert isinstance(res_list, list) and len(res_list) == 2

    response = cast(
        Response,
        asyncio.run(
            client.arun_batch(
                ["Why are there pinenuts?", "Why are there cocoa?"],
                return_response=True,
            )
        ),
    )
    assert (
        isinstance(response.get_response(), list) and len(response.get_response()) == 2
    )
    assert (
        "usage" in response.get_json_response()
        and len(response.get_json_response()["usage"]) == 2
    )
    assert response.get_json_response()["usage"][0]["total_tokens"] == 17
    assert response.get_json_response()["usage"][1]["total_tokens"] == 15

    response = cast(
        Response, client.run("Why are there oranges?", return_response=True)
    )
    assert response.is_cached() is True


@pytest.mark.skipif(not OPENAI_ALIVE, reason="No openai key set")
@pytest.mark.usefixtures("sqlite_cache")
def test_openaichat(sqlite_cache: str) -> None:
    """Test openaichat client."""
    client = Manifest(
        client_name="openaichat",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )

    res = client.run("Why are there apples?")
    assert isinstance(res, str) and len(res) > 0

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert isinstance(response.get_response(), str) and len(response.get_response()) > 0
    assert response.is_cached() is True
    assert "usage" in response.get_json_response()
    assert response.get_json_response()["usage"][0]["total_tokens"] == 22

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert response.is_cached() is True

    response = cast(
        Response, client.run("Why are there oranges?", return_response=True)
    )
    assert response.is_cached() is False

    res_list = asyncio.run(
        client.arun_batch(["Why are there pears?", "Why are there oranges?"])
    )
    assert isinstance(res_list, list) and len(res_list) == 2

    response = cast(
        Response,
        asyncio.run(
            client.arun_batch(
                ["Why are there pinenuts?", "Why are there cocoa?"],
                return_response=True,
            )
        ),
    )
    assert (
        isinstance(response.get_response(), list) and len(response.get_response()) == 2
    )
    assert (
        "usage" in response.get_json_response()
        and len(response.get_json_response()["usage"]) == 2
    )
    assert response.get_json_response()["usage"][0]["total_tokens"] == 24
    assert response.get_json_response()["usage"][1]["total_tokens"] == 22

    response = cast(
        Response, client.run("Why are there oranges?", return_response=True)
    )
    assert response.is_cached() is True
