"""Response test."""
import pytest

from manifest import Response


def test_init():
    """Test response initialization."""
    with pytest.raises(ValueError) as exc_info:
        response = Response(4, False, {})
    assert str(exc_info.value) == "Response must be str or dict. Response is\n4."
    with pytest.raises(ValueError) as exc_info:
        response = Response({"test": "hello"}, False, {})
    assert str(exc_info.value) == (
        "Response must be serialized to a dict with a list of choices. "
        "Response is\n{'test': 'hello'}."
    )
    with pytest.raises(ValueError) as exc_info:
        response = Response({"choices": [{"blah": "hello"}]}, False, {})
    assert str(exc_info.value) == (
        "Response must be serialized to a dict "
        "with a list of choices with text field"
    )

    response = Response({"choices": [{"text": "hello"}]}, False, {})
    assert response._response == {"choices": [{"text": "hello"}]}
    assert response._cached is False
    assert response._request_params == {}

    response = Response({"choices": [{"text": "hello"}]}, True, {"request": "yoyo"})
    assert response._response == {"choices": [{"text": "hello"}]}
    assert response._cached is True
    assert response._request_params == {"request": "yoyo"}


def test_getters():
    """Test response cached."""
    response = Response({"choices": [{"text": "hello"}]}, False, {})
    assert response.get_json_response() == {"choices": [{"text": "hello"}]}
    assert response.is_cached() is False
    assert response.get_request() == {}

    response = Response({"choices": [{"text": "hello"}]}, True, {"request": "yoyo"})
    assert response.get_json_response() == {"choices": [{"text": "hello"}]}
    assert response.is_cached() is True
    assert response.get_request() == {"request": "yoyo"}


def test_serialize():
    """Test response serialization."""
    response = Response({"choices": [{"text": "hello"}]}, True, {"request": "yoyo"})
    deserialized_response = Response.deserialize(response.serialize())
    assert deserialized_response._response == {"choices": [{"text": "hello"}]}
    assert deserialized_response.is_cached() is True
    assert deserialized_response._request_params == {"request": "yoyo"}


def test_get_results():
    """Test response get results."""
    response = Response({"choices": []}, True, {"request": "yoyo"})
    assert response.get_response() is None
    assert response.get_response(stop_token="ll") is None
    assert response.get_response(stop_token="ll", is_batch=True) is None

    response = Response({"choices": [{"text": "hello"}]}, True, {"request": "yoyo"})
    assert response.get_response() == "hello"
    assert response.get_response(stop_token="ll") == "he"
    assert response.get_response(stop_token="ll", is_batch=True) == ["he"]

    response = Response(
        {"choices": [{"text": "hello"}, {"text": "my"}, {"text": "name"}]},
        True,
        {"request": "yoyo"},
    )
    assert response.get_response() == ["hello", "my", "name"]
    assert response.get_response(stop_token="m") == ["hello", "", "na"]
    assert response.get_response(stop_token="m", is_batch=True) == ["hello", "", "na"]
