"""Response test."""
import json

import pytest

from manifest.clients import Response


def test_init():
    """Test response initialization."""
    with pytest.raises(ValueError) as exc_info:
        response = Response(4)
    assert str(exc_info.value) == "Response must be str or dict"
    with pytest.raises(ValueError) as exc_info:
        response = Response({"test": "hello"})
    assert (
        str(exc_info.value)
        == "Response must be serialized to a dict with a list of choices"
    )
    with pytest.raises(ValueError) as exc_info:
        response = Response({"choices": [{"blah": "hello"}]})
    assert str(exc_info.value) == (
        "Response must be serialized to a dict ",
        "with a list of choices with text field",
    )

    response = Response({"choices": [{"text": "hello"}]})
    assert response.response == {"choices": [{"text": "hello"}]}

    response = Response(json.dumps({"choices": [{"text": "hello"}]}))
    assert response.response == {"choices": [{"text": "hello"}]}


def test_getitem():
    """Test response getitem."""
    response = Response({"choices": [{"text": "hello"}]})
    assert response["choices"] == [{"text": "hello"}]


def test_serialize():
    """Test response serialization."""
    response = Response({"choices": [{"text": "hello"}]})
    assert Response.deserialize(response.serialize()).response == {
        "choices": [{"text": "hello"}]
    }


def test_get_results():
    """Test response get results."""
    response = Response({"choices": []})
    assert response.get_results() is None

    response = Response({"choices": [{"text": "hello"}]})
    assert response.get_results() == "hello"

    response = Response(
        {"choices": [{"text": "hello"}, {"text": "my"}, {"text": "name"}]}
    )
    assert response.get_results() == ["hello", "my", "name"]
