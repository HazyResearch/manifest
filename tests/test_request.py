"""Request test."""
from manifest.request import DiffusionRequest, LMRequest


def test_llm_init() -> None:
    """Test request initialization."""
    request = LMRequest()
    assert request.temperature == 0.7

    request = LMRequest(temperature=0.5)
    assert request.temperature == 0.5

    request = LMRequest(**{"temperature": 0.5})  # type: ignore
    assert request.temperature == 0.5

    request = LMRequest(**{"temperature": 0.5, "prompt": "test"})  # type: ignore
    assert request.temperature == 0.5
    assert request.prompt == "test"


def test_diff_init() -> None:
    """Test request initialization."""
    request = DiffusionRequest()
    assert request.height == 512

    request = DiffusionRequest(height=128)
    assert request.height == 128

    request = DiffusionRequest(**{"height": 128})  # type: ignore
    assert request.height == 128

    request = DiffusionRequest(**{"height": 128, "prompt": "test"})  # type: ignore
    assert request.height == 128
    assert request.prompt == "test"


def test_to_dict() -> None:
    """Test request to dict."""
    request_lm = LMRequest()
    dct = request_lm.to_dict()

    assert dct == {k: v for k, v in request_lm.dict().items() if v is not None}

    # Note the second value is a placeholder for the default value
    # It's unused in to_dict
    keys = {"temperature": ("temp", 0.7)}
    dct = request_lm.to_dict(allowable_keys=keys)
    assert dct == {"temp": 0.7, "prompt": ""}

    dct = request_lm.to_dict(allowable_keys=keys, add_prompt=False)
    assert dct == {"temp": 0.7}

    request_diff = DiffusionRequest()
    dct = request_diff.to_dict()

    assert dct == {k: v for k, v in request_diff.dict().items() if v is not None}

    keys = {"height": ("hgt", 512)}
    dct = request_diff.to_dict(allowable_keys=keys)
    assert dct == {"hgt": 512, "prompt": ""}

    dct = request_diff.to_dict(allowable_keys=keys, add_prompt=False)
    assert dct == {"hgt": 512}
