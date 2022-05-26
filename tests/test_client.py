"""
Test client.

We just test the dummy client as we don't want to load a model or use OpenAI tokens.
"""

from manifest.clients.dummy import DummyClient


def test_init():
    """Test client initialization."""
    args = {"num_results": 3}
    client = DummyClient(connection_str=None, client_args=args)
    assert client.num_results == 3


def test_get_request():
    """Test client get request."""
    args = {"num_results": 3}
    client = DummyClient(connection_str=None, client_args=args)
    request_func, request_params = client.get_request("hello")
    assert request_params == {"prompt": "hello", "num_results": 3}
    assert request_func() == {"choices": [{"text": "hello"}] * 3}
