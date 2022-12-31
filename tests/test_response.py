"""Response test."""
import numpy as np
import pytest

from manifest import Response


def test_init() -> None:
    """Test response initialization."""
    with pytest.raises(ValueError) as exc_info:
        response = Response(4, False, {})  # type: ignore
    assert str(exc_info.value) == "Response must be dict. Response is\n4."
    with pytest.raises(ValueError) as exc_info:
        response = Response({"test": "hello"}, False, {})
    assert str(exc_info.value) == (
        "Response must be serialized to a dict with a nonempty list of choices. "
        "Response is\n{'test': 'hello'}."
    )
    with pytest.raises(ValueError) as exc_info:
        response = Response({"choices": [{"blah": "hello"}]}, False, {})
    assert str(exc_info.value) == (
        "Response must be serialized to a dict "
        "with a list of choices with text field"
    )
    with pytest.raises(ValueError) as exc_info:
        response = Response({"choices": []}, False, {})
    assert str(exc_info.value) == (
        "Response must be serialized to a dict with a nonempty list of choices. "
        "Response is\n{'choices': []}."
    )

    response = Response({"choices": [{"text": "hello"}]}, False, {})
    assert response._response == {"choices": [{"text": "hello"}]}
    assert response._cached is False
    assert response._request_params == {}
    assert response.item_dtype is None

    response = Response({"choices": [{"text": "hello"}]}, True, {"request": "yoyo"})
    assert response._response == {"choices": [{"text": "hello"}]}
    assert response._cached is True
    assert response._request_params == {"request": "yoyo"}
    assert response.item_dtype is None

    response = Response(
        {"generations": [{"txt": "hello"}], "logits": []},
        False,
        {},
        generation_key="generations",
        logits_key="logits",
        item_key="txt",
    )
    assert response._response == {"generations": [{"txt": "hello"}], "logits": []}
    assert response._cached is False
    assert response._request_params == {}
    assert response.item_dtype is None

    int_arr = np.random.randint(20, size=(4, 4))
    response = Response(
        {"choices": [{"array": int_arr}]}, True, {"request": "yoyo"}, item_key="array"
    )
    assert response._response == {"choices": [{"array": int_arr}]}
    assert response._cached is True
    assert response._request_params == {"request": "yoyo"}
    assert response.item_dtype == "int64"


def test_getters() -> None:
    """Test response cached."""
    response = Response({"choices": [{"text": "hello"}]}, False, {})
    assert response.get_json_response() == {"choices": [{"text": "hello"}]}
    assert response.is_cached() is False
    assert response.get_request() == {}

    response = Response({"choices": [{"text": "hello"}]}, True, {"request": "yoyo"})
    assert response.get_json_response() == {"choices": [{"text": "hello"}]}
    assert response.is_cached() is True
    assert response.get_request() == {"request": "yoyo"}

    int_arr = np.random.randint(20, size=(4, 4))
    response = Response(
        {"choices": [{"array": int_arr}]}, True, {"request": "yoyo"}, item_key="array"
    )
    assert response.get_json_response() == {"choices": [{"array": int_arr}]}
    assert response.is_cached() is True
    assert response.get_request() == {"request": "yoyo"}


def test_serialize() -> None:
    """Test response serialization."""
    response = Response({"choices": [{"text": "hello"}]}, True, {"request": "yoyo"})
    deserialized_response = Response.deserialize(response.serialize())
    assert deserialized_response._response == {"choices": [{"text": "hello"}]}
    assert deserialized_response.is_cached() is True
    assert deserialized_response._request_params == {"request": "yoyo"}

    int_arr = np.random.randint(20, size=(4, 4))
    response = Response(
        {"choices": [{"array": int_arr}]}, True, {"request": "yoyo"}, item_key="array"
    )
    deserialized_response = Response.deserialize(response.serialize())
    assert np.array_equal(
        deserialized_response._response["choices"][0]["array"], int_arr
    )
    assert deserialized_response.is_cached() is True
    assert deserialized_response._request_params == {"request": "yoyo"}

    float_arr = np.random.randn(4, 4)
    response = Response(
        {"choices": [{"array": float_arr}]}, True, {"request": "yoyo"}, item_key="array"
    )
    deserialized_response = Response.deserialize(response.serialize())
    assert np.array_equal(
        deserialized_response._response["choices"][0]["array"], float_arr
    )
    assert deserialized_response.is_cached() is True
    assert deserialized_response._request_params == {"request": "yoyo"}


def test_get_results() -> None:
    """Test response get results."""
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

    float_arr = np.random.randn(4, 4)
    response = Response(
        {"choices": [{"array": float_arr}, {"array": float_arr}]},
        True,
        {"request": "yoyo"},
        item_key="array",
    )
    assert response.get_response() == [float_arr, float_arr]
    assert response.get_response(stop_token="m") == [float_arr, float_arr]
