"""Request test."""
from manifest.request import DiffusionRequest, LMRequest


def test_llm_init():
    """Test request initialization."""
    request = LMRequest()
    assert request.temperature == 0.7

    request = LMRequest(temperature=0.5)
    assert request.temperature == 0.5

    request = LMRequest(**{"temperature": 0.5})
    assert request.temperature == 0.5

    request = LMRequest(**{"temperature": 0.5, "prompt": "test"})
    assert request.temperature == 0.5
    assert request.prompt == "test"


def test_diff_init():
    """Test request initialization."""
    request = DiffusionRequest()
    assert request.height == 512

    request = DiffusionRequest(height=128)
    assert request.height == 128

    request = DiffusionRequest(**{"height": 128})
    assert request.height == 128

    request = DiffusionRequest(**{"height": 128, "prompt": "test"})
    assert request.height == 128
    assert request.prompt == "test"


def test_to_dict():
    """Test request to dict."""
    request = LMRequest()
    dct = request.to_dict()

    assert dct == {k: v for k, v in request.dict().items() if v is not None}

    # Note the second value is a placeholder for the default value
    # It's unused in to_dict
    keys = {"temperature": ("temp", 0.7)}
    dct = request.to_dict(allowable_keys=keys)
    assert dct == {"temp": 0.7, "prompt": ""}

    dct = request.to_dict(allowable_keys=keys, add_prompt=False)
    assert dct == {"temp": 0.7}

    request = DiffusionRequest()
    dct = request.to_dict()

    assert dct == {k: v for k, v in request.dict().items() if v is not None}

    keys = {"height": ("hgt", 512)}
    dct = request.to_dict(allowable_keys=keys)
    assert dct == {"hgt": 512, "prompt": ""}

    dct = request.to_dict(allowable_keys=keys, add_prompt=False)
    assert dct == {"hgt": 512}
