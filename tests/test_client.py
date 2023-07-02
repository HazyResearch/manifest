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
    assert client.get_model_params() == {
        "engine": "dummy",
        "model": "text-davinci-003",
    }
    assert client.get_model_inputs() == [
        "engine",
        "temperature",
        "max_tokens",
        "n",
        "top_p",
        "top_k",
        "batch_size",
    ]


def test_get_request() -> None:
    """Test client get request."""
    args = {"n": 3}
    client = DummyClient(connection_str=None, client_args=args)
    request_params = client.get_request("hello", {})
    response = client.run_request(request_params)
    assert client.get_cache_key(request_params) == {
        "prompt": "hello",
        "model": "text-davinci-003",
        "n": 3,
        "temperature": 0.0,
        "max_tokens": 10,
        "top_p": 1.0,
        "best_of": 1,
        "engine": "dummy",
        "request_cls": "LMRequest",
    }
    assert response.get_json_response() == {
        "choices": [
            {
                "text": " probsuib.FirstName>- commodityting segunda inserted signals Religious",  # noqa: E501
                "token_logprobs": [
                    -0.2649905035732101,
                    -1.210794839387105,
                    -1.2173929801003434,
                    -0.7758233850171001,
                    -0.7165940659570416,
                    -1.7430328887209088,
                    -1.5379414228820203,
                    -1.7838011423472508,
                    -1.139095076944217,
                    -0.6321855879833425,
                ],
                "tokens": [
                    "70470",
                    "80723",
                    "52693",
                    "39743",
                    "38983",
                    "1303",
                    "56072",
                    "22306",
                    "17738",
                    "53176",
                ],
            }
        ]
        * 3
    }
    assert response.get_usage_obj().dict() == {
        "usages": [{"prompt_tokens": 1, "completion_tokens": 10, "total_tokens": 11}]
        * 3,
    }

    request_params = client.get_request("hello", {"n": 5})
    response = client.run_request(request_params)
    assert client.get_cache_key(request_params) == {
        "prompt": "hello",
        "model": "text-davinci-003",
        "n": 5,
        "temperature": 0.0,
        "max_tokens": 10,
        "top_p": 1.0,
        "best_of": 1,
        "engine": "dummy",
        "request_cls": "LMRequest",
    }
    assert response.get_json_response() == {
        "choices": [
            {
                "text": " probsuib.FirstName>- commodityting segunda inserted signals Religious",  # noqa: E501
                "token_logprobs": [
                    -0.2649905035732101,
                    -1.210794839387105,
                    -1.2173929801003434,
                    -0.7758233850171001,
                    -0.7165940659570416,
                    -1.7430328887209088,
                    -1.5379414228820203,
                    -1.7838011423472508,
                    -1.139095076944217,
                    -0.6321855879833425,
                ],
                "tokens": [
                    "70470",
                    "80723",
                    "52693",
                    "39743",
                    "38983",
                    "1303",
                    "56072",
                    "22306",
                    "17738",
                    "53176",
                ],
            }
        ]
        * 5
    }
    assert response.get_usage_obj().dict() == {
        "usages": [{"prompt_tokens": 1, "completion_tokens": 10, "total_tokens": 11}]
        * 5,
    }

    request_params = client.get_request(["hello"] * 5, {"n": 1})
    response = client.run_request(request_params)
    assert client.get_cache_key(request_params) == {
        "prompt": ["hello"] * 5,
        "model": "text-davinci-003",
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 10,
        "top_p": 1.0,
        "best_of": 1,
        "engine": "dummy",
        "request_cls": "LMRequest",
    }
    assert response.get_json_response() == {
        "choices": [
            {
                "text": " probsuib.FirstName>- commodityting segunda inserted signals Religious",  # noqa: E501
                "token_logprobs": [
                    -0.2649905035732101,
                    -1.210794839387105,
                    -1.2173929801003434,
                    -0.7758233850171001,
                    -0.7165940659570416,
                    -1.7430328887209088,
                    -1.5379414228820203,
                    -1.7838011423472508,
                    -1.139095076944217,
                    -0.6321855879833425,
                ],
                "tokens": [
                    "70470",
                    "80723",
                    "52693",
                    "39743",
                    "38983",
                    "1303",
                    "56072",
                    "22306",
                    "17738",
                    "53176",
                ],
            }
        ]
        * 5
    }
    assert response.get_usage_obj().dict() == {
        "usages": [{"prompt_tokens": 1, "completion_tokens": 10, "total_tokens": 11}]
        * 5,
    }
