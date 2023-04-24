"""
Test client.

We just test the dummy client.
"""
from manifest.clients.dummy import DummyClient


def test_init() -> None:
    """Test client initialization."""
    client = DummyClient(connection_str=None)
    assert client.n == 1  # type: ignore

    args = {"n": 3}
    client = DummyClient(connection_str=None, client_args=args)
    assert client.n == 3  # type: ignore


def test_get_params() -> None:
    """Test get param functions."""
    client = DummyClient(connection_str=None)
    assert client.get_model_params() == {"engine": "dummy"}
    assert client.get_model_inputs() == ["n"]


def test_get_request() -> None:
    """Test client get request."""
    args = {"n": 3}
    client = DummyClient(connection_str=None, client_args=args)
    request_params = client.get_request("hello", {})
    response = client.run_request(request_params)
    assert client.get_cache_key(request_params) == {
        "prompt": "hello",
        "num_results": 3,
        "engine": "dummy",
        "request_cls": "LMRequest",
    }
    assert response.get_json_response() == {
        "choices": [{"text": "hello", "token_logprobs": None, "tokens": None}] * 3,
    }
    assert response.get_usage_obj().dict() == {
        "usages": [{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}] * 3,
    }

    request_params = client.get_request("hello", {"n": 5})
    response = client.run_request(request_params)
    assert client.get_cache_key(request_params) == {
        "prompt": "hello",
        "num_results": 5,
        "engine": "dummy",
        "request_cls": "LMRequest",
    }
    assert response.get_json_response() == {
        "choices": [{"text": "hello", "token_logprobs": None, "tokens": None}] * 5,
    }
    assert response.get_usage_obj().dict() == {
        "usages": [{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}] * 5,
    }

    request_params = client.get_request(["hello"] * 5, {"n": 1})
    response = client.run_request(request_params)
    assert client.get_cache_key(request_params) == {
        "prompt": ["hello"] * 5,
        "num_results": 1,
        "engine": "dummy",
        "request_cls": "LMRequest",
    }
    assert response.get_json_response() == {
        "choices": [{"text": "hello", "token_logprobs": None, "tokens": None}] * 5,
    }
    assert response.get_usage_obj().dict() == {
        "usages": [{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}] * 5,
    }
