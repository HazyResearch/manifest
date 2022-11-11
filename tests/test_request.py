"""Request test."""
from manifest import Request


def test_init():
    """Test request initialization."""
    request = Request()
    assert request.temperature == 0.7

    request = Request(temperature=0.5)
    assert request.temperature == 0.5

    request = Request(**{"temperature": 0.5})
    assert request.temperature == 0.5

    request = Request(**{"temperature": 0.5, "prompt": "test"})
    assert request.temperature == 0.5
    assert request.prompt == "test"


def test_to_dict():
    """Test request to dict."""
    request = Request()
    dct = request.to_dict()

    assert dct == {k: v for k, v in request.dict().items() if v is not None}

    keys = {"temperature": ("temp", 0.5)}
    dct = request.to_dict(allowable_keys=keys)
    assert dct == {"temp": 0.7, "prompt": ""}

    dct = request.to_dict(allowable_keys=keys, add_prompt=False)
    assert dct == {"temp": 0.7}
