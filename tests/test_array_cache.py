"""Array cache test."""
from pathlib import Path

import numpy as np
import pytest

from manifest.caches.array_cache import ArrayCache


def test_init(tmpdir: Path) -> None:
    """Test cache initialization."""
    cache = ArrayCache(Path(tmpdir))
    assert (tmpdir / "hash2arrloc.sqlite").exists()
    assert cache.cur_file_idx == 0
    assert cache.cur_offset == 0


def test_put_get(tmpdir: Path) -> None:
    """Test putting and getting."""
    cache = ArrayCache(tmpdir)
    cache.max_memmap_size = 5
    arr = np.random.rand(10, 10)

    with pytest.raises(ValueError) as exc_info:
        cache.put("key", arr)
    assert str(exc_info.value) == ("Array is too large to be cached. Max is 5")

    cache.max_memmap_size = 120
    cache.put("key", arr)
    assert np.allclose(cache.get("key"), arr)
    assert cache.get("key").dtype == arr.dtype
    assert cache.cur_file_idx == 0
    assert cache.cur_offset == 100
    assert cache.hash2arrloc["key"] == {
        "file_idx": 0,
        "offset": 0,
        "flatten_size": 100,
        "shape": (10, 10),
        "dtype": np.dtype("float64"),
    }

    arr2 = np.random.randint(0, 3, size=(10, 10))
    cache.put("key2", arr2)
    assert np.allclose(cache.get("key2"), arr2)
    assert cache.get("key2").dtype == arr2.dtype
    assert cache.cur_file_idx == 1
    assert cache.cur_offset == 100
    assert cache.hash2arrloc["key2"] == {
        "file_idx": 1,
        "offset": 0,
        "flatten_size": 100,
        "shape": (10, 10),
        "dtype": np.dtype("int64"),
    }

    cache = ArrayCache(tmpdir)
    assert cache.hash2arrloc["key"] == {
        "file_idx": 0,
        "offset": 0,
        "flatten_size": 100,
        "shape": (10, 10),
        "dtype": np.dtype("float64"),
    }
    assert cache.hash2arrloc["key2"] == {
        "file_idx": 1,
        "offset": 0,
        "flatten_size": 100,
        "shape": (10, 10),
        "dtype": np.dtype("int64"),
    }
    assert np.allclose(cache.get("key"), arr)
    assert np.allclose(cache.get("key2"), arr2)


def test_contains_key(tmpdir: Path) -> None:
    """Test contains key."""
    cache = ArrayCache(tmpdir)
    assert not cache.contains_key("key")
    arr = np.random.rand(10, 10)
    cache.put("key", arr)
    assert cache.contains_key("key")
