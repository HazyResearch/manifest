"""Manifest test."""
import asyncio
import os
from typing import Iterator, cast
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import requests
from requests import HTTPError

from manifest import Manifest, Response
from manifest.caches.noop import NoopCache
from manifest.caches.sqlite import SQLiteCache
from manifest.clients.dummy import DummyClient
from manifest.connections.client_pool import ClientConnection

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
    assert len(manifest.client_pool.client_pool) == 1
    client = manifest.client_pool.get_next_client()
    assert isinstance(client, DummyClient)
    assert isinstance(manifest.cache, SQLiteCache)
    assert client.n == 1  # type: ignore
    assert manifest.stop_token == ""

    manifest = Manifest(
        client_name="dummy",
        cache_name="noop",
        n=3,
        stop_token="\n",
    )
    assert len(manifest.client_pool.client_pool) == 1
    client = manifest.client_pool.get_next_client()
    assert isinstance(client, DummyClient)
    assert isinstance(manifest.cache, NoopCache)
    assert client.n == 3  # type: ignore
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
        temperature=0.0,
    )

    prompt = "This is a prompt"
    with pytest.raises(ValueError) as exc_info:
        result = manifest.run(prompt, return_response=return_response, bad_input=5)
    assert str(exc_info.value) == "[('bad_input', 5)] arguments are not recognized."

    result = manifest.run(prompt, return_response=return_response, top_k=5)
    assert result is not None

    prompt = "This is a prompt"
    result = manifest.run(prompt, return_response=return_response)
    if return_response:
        assert isinstance(result, Response)
        result = cast(Response, result)
        assert len(result.get_usage_obj().usages) == len(
            result.get_response_obj().choices
        )
        res = result.get_response(manifest.stop_token)
    else:
        res = cast(str, result)

    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": n,
                "prompt": "This is a prompt",
                "request_cls": "LMRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is not None
    )
    if n == 1:
        assert res == "Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines"
    else:
        assert res == [
            "Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines",
            "Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines",
        ]

    prompt = "This is a prompt"
    result = manifest.run(prompt, run_id="34", return_response=return_response)
    if return_response:
        assert isinstance(result, Response)
        result = cast(Response, result)
        assert len(result.get_usage_obj().usages) == len(
            result.get_response_obj().choices
        )
        res = result.get_response(manifest.stop_token)
    else:
        res = cast(str, result)
    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": n,
                "prompt": "This is a prompt",
                "request_cls": "LMRequest",
                "temperature": 0.0,
                "top_p": 1.0,
                "run_id": "34",
            }
        )
        is not None
    )
    if n == 1:
        assert res == "Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines"
    else:
        assert res == [
            "Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines",
            "Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines",
        ]

    prompt = "Hello is a prompt"
    result = manifest.run(prompt, return_response=return_response)
    if return_response:
        assert isinstance(result, Response)
        result = cast(Response, result)
        assert len(result.get_usage_obj().usages) == len(
            result.get_response_obj().choices
        )
        res = result.get_response(manifest.stop_token)
    else:
        res = cast(str, result)
    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": n,
                "prompt": "Hello is a prompt",
                "request_cls": "LMRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is not None
    )
    if n == 1:
        assert res == "appersstoff210 currentNodeleh norm unified_voice DIYHam"
    else:
        assert res == [
            "appersstoff210 currentNodeleh norm unified_voice DIYHam",
            "appersstoff210 currentNodeleh norm unified_voice DIYHam",
        ]

    prompt = "Hello is a prompt"
    result = manifest.run(
        prompt, stop_token=" current", return_response=return_response
    )
    if return_response:
        assert isinstance(result, Response)
        result = cast(Response, result)
        assert len(result.get_usage_obj().usages) == len(
            result.get_response_obj().choices
        )
        res = result.get_response(stop_token=" current")
    else:
        res = cast(str, result)
    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": n,
                "prompt": "Hello is a prompt",
                "request_cls": "LMRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is not None
    )
    if n == 1:
        assert res == "appersstoff210"
    else:
        assert res == ["appersstoff210", "appersstoff210"]


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
        temperature=0.0,
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
            assert len(result.get_usage_obj().usages) == len(
                result.get_response_obj().choices
            )
            res = result.get_response(manifest.stop_token, is_batch=True)
        else:
            res = cast(str, result)
        assert res == ["Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines"]
        assert (
            manifest.cache.get(
                {
                    "best_of": 1,
                    "engine": "dummy",
                    "max_tokens": 10,
                    "model": "text-davinci-003",
                    "n": n,
                    "prompt": "This is a prompt",
                    "request_cls": "LMRequest",
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
            )
            is not None
        )

        prompt = ["Hello is a prompt", "Hello is a prompt"]
        result = manifest.run(prompt, return_response=return_response)
        if return_response:
            assert isinstance(result, Response)
            result = cast(Response, result)
            assert len(result.get_usage_obj().usages) == len(
                result.get_response_obj().choices
            )
            res = result.get_response(manifest.stop_token, is_batch=True)
        else:
            res = cast(str, result)
        assert res == [
            "appersstoff210 currentNodeleh norm unified_voice DIYHam",
            "appersstoff210 currentNodeleh norm unified_voice DIYHam",
        ]
        assert (
            manifest.cache.get(
                {
                    "best_of": 1,
                    "engine": "dummy",
                    "max_tokens": 10,
                    "model": "text-davinci-003",
                    "n": n,
                    "prompt": "Hello is a prompt",
                    "request_cls": "LMRequest",
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
            )
            is not None
        )

        result = manifest.run(prompt, return_response=True)
        res = cast(Response, result).get_response(manifest.stop_token, is_batch=True)
        assert cast(Response, result).is_cached()

        assert (
            manifest.cache.get(
                {
                    "best_of": 1,
                    "engine": "dummy",
                    "max_tokens": 10,
                    "model": "text-davinci-003",
                    "n": n,
                    "prompt": "New prompt",
                    "request_cls": "LMRequest",
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
            )
            is None
        )
        prompt = ["This is a prompt", "New prompt"]
        result = manifest.run(prompt, return_response=return_response)
        if return_response:
            assert isinstance(result, Response)
            result = cast(Response, result)
            assert len(result.get_usage_obj().usages) == len(
                result.get_response_obj().choices
            )
            res = result.get_response(manifest.stop_token, is_batch=True)
            # Cached because one item is in cache
            assert result.is_cached()
        else:
            res = cast(str, result)
        assert res == [
            "Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines",
            ".vol.deserializebigmnchantment ROTıl='')\najsС",
        ]

        prompt = ["Hello is a prompt", "Hello is a prompt"]
        result = manifest.run(
            prompt, stop_token=" current", return_response=return_response
        )
        if return_response:
            assert isinstance(result, Response)
            result = cast(Response, result)
            assert len(result.get_usage_obj().usages) == len(
                result.get_response_obj().choices
            )
            res = result.get_response(stop_token=" current", is_batch=True)
        else:
            res = cast(str, result)
        assert res == ["appersstoff210", "appersstoff210"]


@pytest.mark.usefixtures("sqlite_cache")
def test_abatch_run(sqlite_cache: str) -> None:
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        temperature=0.0,
    )
    prompt = ["This is a prompt"]
    result = cast(
        Response, asyncio.run(manifest.arun_batch(prompt, return_response=True))
    )

    assert len(result.get_usage_obj().usages) == len(result.get_response_obj().choices)
    res = result.get_response(manifest.stop_token, is_batch=True)
    assert res == ["Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines"]
    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": 1,
                "prompt": "This is a prompt",
                "request_cls": "LMRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is not None
    )

    prompt = ["Hello is a prompt", "Hello is a prompt"]
    result = cast(
        Response, asyncio.run(manifest.arun_batch(prompt, return_response=True))
    )

    assert len(result.get_usage_obj().usages) == len(result.get_response_obj().choices)
    res = result.get_response(manifest.stop_token, is_batch=True)
    assert res == [
        "appersstoff210 currentNodeleh norm unified_voice DIYHam",
        "appersstoff210 currentNodeleh norm unified_voice DIYHam",
    ]
    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": 1,
                "prompt": "Hello is a prompt",
                "request_cls": "LMRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is not None
    )

    result = cast(
        Response, asyncio.run(manifest.arun_batch(prompt, return_response=True))
    )

    assert len(result.get_usage_obj().usages) == len(result.get_response_obj().choices)
    res = result.get_response(manifest.stop_token, is_batch=True)
    assert result.is_cached()

    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": 1,
                "prompt": "New prompt",
                "request_cls": "LMRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is None
    )
    prompt = ["This is a prompt", "New prompt"]
    result = cast(
        Response, asyncio.run(manifest.arun_batch(prompt, return_response=True))
    )

    assert len(result.get_usage_obj().usages) == len(result.get_response_obj().choices)
    res = result.get_response(manifest.stop_token, is_batch=True)
    # Cached because one item is in cache
    assert result.is_cached()
    assert res == [
        "Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines",
        ".vol.deserializebigmnchantment ROTıl='')\najsС",
    ]

    prompt = ["Hello is a prompt", "Hello is a prompt"]
    result = cast(
        Response, asyncio.run(manifest.arun_batch(prompt, return_response=True))
    )

    assert len(result.get_usage_obj().usages) == len(result.get_response_obj().choices)
    res = result.get_response(stop_token=" current", is_batch=True)
    assert res == ["appersstoff210", "appersstoff210"]


@pytest.mark.usefixtures("sqlite_cache")
def test_run_chat(sqlite_cache: str) -> None:
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        temperature=0.0,
    )
    # Set CHAT to be true for this model
    manifest.client_pool.client_pool[0].IS_CHAT = True

    prompt = [
        {"role": "system", "content": "Hello."},
    ]
    result = manifest.run(prompt, return_response=False)
    assert (
        result
        == "ectors WortGo ré_sg|--------------------------------------------------------------------------\n contradictory Aad \u200b getUserId"  # noqa: E501
    )
    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": 1,
                "prompt": [{"content": "Hello.", "role": "system"}],
                "request_cls": "LMChatRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is not None
    )

    prompt = [
        {"role": "system", "content": "Hello."},
        {"role": "user", "content": "Goodbye?"},
    ]
    result = manifest.run(prompt, return_response=True)
    assert isinstance(result, Response)
    result = cast(Response, result)
    assert len(result.get_usage_obj().usages) == len(result.get_response_obj().choices)
    res = result.get_response()
    assert res == "_deploy_age_gp hora Plus Scheduler EisenhowerRF视 chemotherapy"
    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": 1,
                "prompt": [
                    {"role": "system", "content": "Hello."},
                    {"role": "user", "content": "Goodbye?"},
                ],
                "request_cls": "LMChatRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is not None
    )


@pytest.mark.usefixtures("sqlite_cache")
def test_score_run(sqlite_cache: str) -> None:
    """Test manifest run."""
    manifest = Manifest(
        client_name="dummy",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        temperature=0.0,
    )

    prompt = "This is a prompt"
    result = manifest.score_prompt(prompt)
    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": 1,
                "prompt": "This is a prompt",
                "request_cls": "LMScoreRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is not None
    )
    assert result == {
        "response": {
            "choices": [
                {
                    "text": "Nice Employ NFCYouryms“Inwarn\ttemplate europ Moines",
                    "token_logprobs": [
                        -1.827188890438529,
                        -1.6981601736417915,
                        -0.24606708391178755,
                        -1.9209383499010613,
                        -0.8833563758318617,
                        -1.4121369466920703,
                        -0.376352908076236,
                        -1.3200064558188096,
                        -0.813028447207917,
                        -0.5977255311239729,
                    ],
                    "tokens": [
                        "46078",
                        "21445",
                        "48305",
                        "7927",
                        "76125",
                        "46233",
                        "34581",
                        "23679",
                        "63021",
                        "78158",
                    ],
                }
            ]
        },
        "usages": {
            "usages": [
                {"completion_tokens": 10, "prompt_tokens": 4, "total_tokens": 14}
            ]
        },
        "cached": False,
        "request": {
            "prompt": "This is a prompt",
            "engine": "text-davinci-003",
            "n": 1,
            "client_timeout": 60,
            "run_id": None,
            "batch_size": 20,
            "temperature": 0.0,
            "max_tokens": 10,
            "top_p": 1.0,
            "top_k": 1,
            "logprobs": None,
            "stop_sequences": None,
            "num_beams": 1,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "response_type": "text",
        "request_type": "LMScoreRequest",
        "item_dtype": None,
    }

    prompt_list = ["Hello is a prompt", "Hello is another prompt"]
    result = manifest.score_prompt(prompt_list)
    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": 1,
                "prompt": "Hello is a prompt",
                "request_cls": "LMScoreRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is not None
    )
    assert (
        manifest.cache.get(
            {
                "best_of": 1,
                "engine": "dummy",
                "max_tokens": 10,
                "model": "text-davinci-003",
                "n": 1,
                "prompt": "Hello is another prompt",
                "request_cls": "LMScoreRequest",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        is not None
    )
    assert result == {
        "response": {
            "choices": [
                {
                    "text": "appersstoff210 currentNodeleh norm unified_voice DIYHam",
                    "token_logprobs": [
                        -0.5613340599860608,
                        -1.2822870706137146,
                        -1.9909319620162806,
                        -0.6312373658222814,
                        -1.9066239705571664,
                        -1.2420939968397082,
                        -0.7208735169940805,
                        -1.9144266963723062,
                        -0.041181937860757856,
                        -0.5356282450367043,
                    ],
                    "tokens": [
                        "28921",
                        "81056",
                        "8848",
                        "47399",
                        "74890",
                        "7617",
                        "43790",
                        "77865",
                        "32558",
                        "41041",
                    ],
                },
                {
                    "text": ".addAttribute_size DE imageUrl_datas\tapFixed(hour setups\tcomment",  # noqa: E501
                    "token_logprobs": [
                        -1.1142500072582333,
                        -0.819706434396527,
                        -1.9956443391600693,
                        -0.8425896744807639,
                        -1.8398050571245623,
                        -1.912564137256891,
                        -1.6677665162080606,
                        -1.1579612203844727,
                        -1.9876114502998343,
                        -0.2698297864722319,
                    ],
                    "tokens": [
                        "26300",
                        "2424",
                        "3467",
                        "40749",
                        "47630",
                        "70998",
                        "13829",
                        "72135",
                        "84823",
                        "97368",
                    ],
                },
            ]
        },
        "usages": {
            "usages": [
                {"completion_tokens": 10, "prompt_tokens": 4, "total_tokens": 14},
                {"completion_tokens": 10, "prompt_tokens": 4, "total_tokens": 14},
            ]
        },
        "cached": False,
        "request": {
            "prompt": ["Hello is a prompt", "Hello is another prompt"],
            "engine": "text-davinci-003",
            "n": 1,
            "client_timeout": 60,
            "run_id": None,
            "batch_size": 20,
            "temperature": 0.0,
            "max_tokens": 10,
            "top_p": 1.0,
            "top_k": 1,
            "logprobs": None,
            "stop_sequences": None,
            "num_beams": 1,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "response_type": "text",
        "request_type": "LMScoreRequest",
        "item_dtype": None,
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


@pytest.mark.skipif(not MODEL_ALIVE, reason=f"No model at {URL}")
@pytest.mark.usefixtures("sqlite_cache")
def test_local_huggingfaceembedding(sqlite_cache: str) -> None:
    """Test openaichat client."""
    client = Manifest(
        client_name="huggingfaceembedding",
        client_connection=URL,
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )

    res = client.run("Why are there carrots?")
    assert isinstance(res, np.ndarray)

    response = cast(
        Response, client.run("Why are there carrots?", return_response=True)
    )
    assert isinstance(response.get_response(), np.ndarray)
    assert np.allclose(response.get_response(), res)

    client = Manifest(
        client_name="huggingfaceembedding",
        client_connection=URL,
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )

    res = client.run("Why are there apples?")
    assert isinstance(res, np.ndarray)

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert isinstance(response.get_response(), np.ndarray)
    assert np.allclose(response.get_response(), res)
    assert response.is_cached() is True

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert response.is_cached() is True

    res_list = client.run(["Why are there apples?", "Why are there bananas?"])
    assert (
        isinstance(res_list, list)
        and len(res_list) == 2
        and isinstance(res_list[0], np.ndarray)
    )

    response = cast(
        Response,
        client.run(
            ["Why are there apples?", "Why are there mangos?"], return_response=True
        ),
    )
    assert (
        isinstance(response.get_response(), list) and len(response.get_response()) == 2
    )

    response = cast(
        Response, client.run("Why are there bananas?", return_response=True)
    )
    assert response.is_cached() is True

    response = cast(
        Response, client.run("Why are there oranges?", return_response=True)
    )
    assert response.is_cached() is False

    res_list = asyncio.run(
        client.arun_batch(["Why are there pears?", "Why are there oranges?"])
    )
    assert (
        isinstance(res_list, list)
        and len(res_list) == 2
        and isinstance(res_list[0], np.ndarray)
    )

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
        isinstance(response.get_response(), list)
        and len(res_list) == 2
        and isinstance(res_list[0], np.ndarray)
    )

    response = cast(
        Response, client.run("Why are there oranges?", return_response=True)
    )
    assert response.is_cached() is True


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
    assert response.get_response() == res
    assert response.is_cached() is True
    assert response.get_usage_obj().usages
    assert response.get_usage_obj().usages[0].total_tokens == 15

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
    assert response.get_usage_obj().usages and len(response.get_usage_obj().usages) == 2
    assert response.get_usage_obj().usages[0].total_tokens == 15
    assert response.get_usage_obj().usages[1].total_tokens == 16

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
    assert response.get_usage_obj().usages and len(response.get_usage_obj().usages) == 2
    assert response.get_usage_obj().usages[0].total_tokens == 17
    assert response.get_usage_obj().usages[1].total_tokens == 15

    response = cast(
        Response, client.run("Why are there oranges?", return_response=True)
    )
    assert response.is_cached() is True

    # Test streaming
    num_responses = 0
    streaming_response_text = cast(
        Iterator[str], client.run("Why are there oranges?", stream=True)
    )
    for res_text in streaming_response_text:
        num_responses += 1
        assert isinstance(res_text, str) and len(res_text) > 0
    assert num_responses == 8

    streaming_response = cast(
        Iterator[Response],
        client.run("Why are there mandarines?", return_response=True, stream=True),
    )
    num_responses = 0
    merged_res = []
    for res in streaming_response:
        num_responses += 1
        assert isinstance(res, Response) and len(res.get_response()) > 0
        merged_res.append(cast(str, res.get_response()))
        assert not res.is_cached()
    assert num_responses == 10

    # Make sure cached
    streaming_response = cast(
        Iterator[Response],
        client.run("Why are there mandarines?", return_response=True, stream=True),
    )
    num_responses = 0
    merged_res_cachced = []
    for res in streaming_response:
        num_responses += 1
        assert isinstance(res, Response) and len(res.get_response()) > 0
        merged_res_cachced.append(cast(str, res.get_response()))
        assert res.is_cached()
    # OpenAI stream does not return logprobs, so this is by number of words
    assert num_responses == 7
    assert "".join(merged_res) == "".join(merged_res_cachced)


@pytest.mark.skipif(not OPENAI_ALIVE, reason="No openai key set")
@pytest.mark.usefixtures("sqlite_cache")
def test_openaichat(sqlite_cache: str) -> None:
    """Test openaichat client."""
    client = Manifest(
        client_name="openaichat",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        temperature=0.0,
    )

    res = client.run("Why are there apples?")
    assert isinstance(res, str) and len(res) > 0

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert isinstance(response.get_response(), str) and len(response.get_response()) > 0
    assert response.get_response() == res
    assert response.is_cached() is True
    assert response.get_usage_obj().usages
    assert response.get_usage_obj().usages[0].total_tokens == 23

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
    assert response.get_usage_obj().usages and len(response.get_usage_obj().usages) == 2
    assert response.get_usage_obj().usages[0].total_tokens == 25
    assert response.get_usage_obj().usages[1].total_tokens == 23

    response = cast(
        Response, client.run("Why are there oranges?", return_response=True)
    )
    assert response.is_cached() is True

    chat_dict = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020.",
        },
        {"role": "user", "content": "Where was it played?"},
    ]
    res = client.run(chat_dict)
    assert isinstance(res, str) and len(res) > 0
    response = cast(Response, client.run(chat_dict, return_response=True))
    assert response.is_cached() is True
    assert response.get_usage_obj().usages[0].total_tokens == 67
    chat_dict = [
        {"role": "system", "content": "You are a helpful assistanttttt."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020.",
        },
        {"role": "user", "content": "Where was it played?"},
    ]
    response = cast(Response, client.run(chat_dict, return_response=True))
    assert response.is_cached() is False

    # Test streaming
    num_responses = 0
    streaming_response_text = cast(
        Iterator[str], client.run("Why are there oranges?", stream=True)
    )
    for res_text in streaming_response_text:
        num_responses += 1
        assert isinstance(res_text, str) and len(res_text) > 0
    assert num_responses == 9

    streaming_response = cast(
        Iterator[Response],
        client.run("Why are there mandarines?", return_response=True, stream=True),
    )
    num_responses = 0
    merged_res = []
    for res in streaming_response:
        num_responses += 1
        assert isinstance(res, Response) and len(res.get_response()) > 0
        merged_res.append(cast(str, res.get_response()))
        assert not res.is_cached()
    assert num_responses == 10

    # Make sure cached
    streaming_response = cast(
        Iterator[Response],
        client.run("Why are there mandarines?", return_response=True, stream=True),
    )
    num_responses = 0
    merged_res_cachced = []
    for res in streaming_response:
        num_responses += 1
        assert isinstance(res, Response) and len(res.get_response()) > 0
        merged_res_cachced.append(cast(str, res.get_response()))
        assert res.is_cached()
    # OpenAI stream does not return logprobs, so this is by number of words
    assert num_responses == 7
    assert "".join(merged_res) == "".join(merged_res_cachced)


@pytest.mark.skipif(not OPENAI_ALIVE, reason="No openai key set")
@pytest.mark.usefixtures("sqlite_cache")
def test_openaiembedding(sqlite_cache: str) -> None:
    """Test openaichat client."""
    client = Manifest(
        client_name="openaiembedding",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
        array_serializer="local_file",
    )

    res = client.run("Why are there carrots?")
    assert isinstance(res, np.ndarray)

    response = cast(
        Response, client.run("Why are there carrots?", return_response=True)
    )
    assert isinstance(response.get_response(), np.ndarray)
    assert np.allclose(response.get_response(), res)

    client = Manifest(
        client_name="openaiembedding",
        cache_name="sqlite",
        cache_connection=sqlite_cache,
    )

    res = client.run("Why are there apples?")
    assert isinstance(res, np.ndarray)

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert isinstance(response.get_response(), np.ndarray)
    assert np.allclose(response.get_response(), res)
    assert response.is_cached() is True
    assert response.get_usage_obj().usages
    assert response.get_usage_obj().usages[0].total_tokens == 5

    response = cast(Response, client.run("Why are there apples?", return_response=True))
    assert response.is_cached() is True

    res_list = client.run(["Why are there apples?", "Why are there bananas?"])
    assert (
        isinstance(res_list, list)
        and len(res_list) == 2
        and isinstance(res_list[0], np.ndarray)
    )

    response = cast(
        Response,
        client.run(
            ["Why are there apples?", "Why are there mangos?"], return_response=True
        ),
    )
    assert (
        isinstance(response.get_response(), list) and len(response.get_response()) == 2
    )
    assert response.get_usage_obj().usages and len(response.get_usage_obj().usages) == 2
    assert response.get_usage_obj().usages[0].total_tokens == 5
    assert response.get_usage_obj().usages[1].total_tokens == 6

    response = cast(
        Response, client.run("Why are there bananas?", return_response=True)
    )
    assert response.is_cached() is True

    response = cast(
        Response, client.run("Why are there oranges?", return_response=True)
    )
    assert response.is_cached() is False

    res_list = asyncio.run(
        client.arun_batch(["Why are there pears?", "Why are there oranges?"])
    )
    assert (
        isinstance(res_list, list)
        and len(res_list) == 2
        and isinstance(res_list[0], np.ndarray)
    )

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
        isinstance(response.get_response(), list)
        and len(res_list) == 2
        and isinstance(res_list[0], np.ndarray)
    )
    assert response.get_usage_obj().usages and len(response.get_usage_obj().usages) == 2
    assert response.get_usage_obj().usages[0].total_tokens == 7
    assert response.get_usage_obj().usages[1].total_tokens == 5

    response = cast(
        Response, client.run("Why are there oranges?", return_response=True)
    )
    assert response.is_cached() is True


@pytest.mark.skipif(not OPENAI_ALIVE, reason="No openai key set")
@pytest.mark.usefixtures("sqlite_cache")
def test_openai_pool(sqlite_cache: str) -> None:
    """Test openai and openaichat client."""
    client_connection1 = ClientConnection(
        client_name="openaichat",
    )
    client_connection2 = ClientConnection(client_name="openai", engine="text-ada-001")
    client = Manifest(
        client_pool=[client_connection1, client_connection2],
        cache_name="sqlite",
        client_connection=sqlite_cache,
    )
    res = client.run("Why are there apples?")
    assert isinstance(res, str) and len(res) > 0

    res2 = client.run("Why are there apples?")
    assert isinstance(res2, str) and len(res2) > 0
    # Different models
    assert res != res2

    assert cast(
        Response, client.run("Why are there apples?", return_response=True)
    ).is_cached()

    res_list = asyncio.run(
        client.arun_batch(["Why are there pears?", "Why are there oranges?"])
    )
    assert isinstance(res_list, list) and len(res_list) == 2
    res_list2 = asyncio.run(
        client.arun_batch(["Why are there pears?", "Why are there oranges?"])
    )
    assert isinstance(res_list2, list) and len(res_list2) == 2
    # Different models
    assert res_list != res_list2

    assert cast(
        Response,
        asyncio.run(
            client.arun_batch(
                ["Why are there pears?", "Why are there oranges?"], return_response=True
            )
        ),
    ).is_cached()

    # Test chunk size of 1
    res_list = asyncio.run(
        client.arun_batch(
            ["Why are there pineapples?", "Why are there pinecones?"], chunk_size=1
        )
    )
    assert isinstance(res_list, list) and len(res_list) == 2
    res_list2 = asyncio.run(
        client.arun_batch(
            ["Why are there pineapples?", "Why are there pinecones?"], chunk_size=1
        )
    )
    # Because we split across both models exactly in first run,
    # we will get the same result
    assert res_list == res_list2


@pytest.mark.skipif(
    not OPENAI_ALIVE or not MODEL_ALIVE, reason="No openai or local model set"
)
@pytest.mark.usefixtures("sqlite_cache")
def test_mixed_pool(sqlite_cache: str) -> None:
    """Test openai and openaichat client."""
    client_connection1 = ClientConnection(
        client_name="huggingface",
        client_connection=URL,
    )
    client_connection2 = ClientConnection(client_name="openai", engine="text-ada-001")
    client = Manifest(
        client_pool=[client_connection1, client_connection2],
        cache_name="sqlite",
        client_connection=sqlite_cache,
    )

    res = client.run("Why are there apples?")
    assert isinstance(res, str) and len(res) > 0

    res2 = client.run("Why are there apples?")
    assert isinstance(res2, str) and len(res2) > 0
    # Different models
    assert res != res2
    assert cast(
        Response, client.run("Why are there apples?", return_response=True)
    ).is_cached()

    res_list = asyncio.run(
        client.arun_batch(["Why are there pears?", "Why are there oranges?"])
    )
    assert isinstance(res_list, list) and len(res_list) == 2
    res_list2 = asyncio.run(
        client.arun_batch(["Why are there pears?", "Why are there oranges?"])
    )
    assert isinstance(res_list2, list) and len(res_list2) == 2
    # Different models
    assert res_list != res_list2

    assert cast(
        Response,
        asyncio.run(
            client.arun_batch(
                ["Why are there pears?", "Why are there oranges?"], return_response=True
            )
        ),
    ).is_cached()

    # Test chunk size of 1
    res_list = asyncio.run(
        client.arun_batch(
            ["Why are there pineapples?", "Why are there pinecones?"], chunk_size=1
        )
    )
    assert isinstance(res_list, list) and len(res_list) == 2
    res_list2 = asyncio.run(
        client.arun_batch(
            ["Why are there pineapples?", "Why are there pinecones?"], chunk_size=1
        )
    )
    # Because we split across both models exactly in first run,
    # we will get the same result
    assert res_list == res_list2


def test_retry_handling() -> None:
    """Test retry handling."""
    # We'll mock the response so we won't need a real connection
    client = Manifest(client_name="openai", client_connection="fake")
    mock_create = MagicMock(
        side_effect=[
            # raise a 429 error
            HTTPError(
                response=Mock(status_code=429, json=Mock(return_value={})),
                request=Mock(),
            ),
            # get a valid http response with a 200 status code
            Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "choices": [
                            {
                                "finish_reason": "length",
                                "index": 0,
                                "logprobs": None,
                                "text": " WHATTT.",
                            },
                            {
                                "finish_reason": "length",
                                "index": 1,
                                "logprobs": None,
                                "text": " UH OH.",
                            },
                            {
                                "finish_reason": "length",
                                "index": 2,
                                "logprobs": None,
                                "text": " HARG",
                            },
                        ],
                        "created": 1679469056,
                        "id": "cmpl-6wmuWfmyuzi68B6gfeNC0h5ywxXL5",
                        "model": "text-ada-001",
                        "object": "text_completion",
                        "usage": {
                            "completion_tokens": 30,
                            "prompt_tokens": 24,
                            "total_tokens": 54,
                        },
                    }
                ),
            ),
        ]
    )
    prompts = [
        "The sky is purple. This is because",
        "The sky is magnet. This is because",
        "The sky is fuzzy. This is because",
    ]
    with patch("manifest.clients.client.requests.post", mock_create):
        # Run manifest
        result = client.run(prompts, temperature=0, overwrite_cache=True)
        assert result == [" WHATTT.", " UH OH.", " HARG"]

        # Assert that OpenAI client was called twice
        assert mock_create.call_count == 2

    # Now make sure it errors when not a 429 or 500
    mock_create = MagicMock(
        side_effect=[
            # raise a 505 error
            HTTPError(
                response=Mock(status_code=505, json=Mock(return_value={})),
                request=Mock(),
            ),
        ]
    )
    with patch("manifest.clients.client.requests.post", mock_create):
        # Run manifest
        with pytest.raises(HTTPError):
            client.run(prompts, temperature=0, overwrite_cache=True)

        # Assert that OpenAI client was called once
        assert mock_create.call_count == 1
