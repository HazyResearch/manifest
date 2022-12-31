"""
Test client.

We just test the dummy client as we don't want to load a model or use OpenAI tokens.
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
    request_params = client.get_request_params("hello", {})
    request_func, request_params_return = client.get_request(request_params)
    assert request_params_return == {"prompt": "hello", "num_results": 3}
    assert request_func() == {"choices": [{"text": "hello"}] * 3}

    request_params = client.get_request_params("hello", {"n": 5})
    request_func, request_params_return = client.get_request(request_params)
    assert request_params_return == {"prompt": "hello", "num_results": 5}
    assert request_func() == {"choices": [{"text": "hello"}] * 5}

    request_params = client.get_request_params(["hello"] * 5, {"n": 1})
    request_func, request_params_return = client.get_request(request_params)
    assert request_params_return == {"prompt": ["hello"] * 5, "num_results": 1}
    assert request_func() == {"choices": [{"text": "hello"}] * 5}
