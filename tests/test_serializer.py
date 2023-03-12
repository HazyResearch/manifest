"""Cache test."""
import json

import numpy as np

from manifest.caches.serializers import ArraySerializer


def test_response_to_key() -> None:
    """Test array serializer initialization."""
    serializer = ArraySerializer()
    arr = np.random.rand(4, 4)
    res = {"choices": [{"array": arr}]}
    key = serializer.response_to_key(res)
    key_dct = json.loads(key)
    assert isinstance(key_dct["choices"][0]["array"], str)

    res2 = serializer.key_to_response(key)
    assert np.allclose(arr, res2["choices"][0]["array"])
