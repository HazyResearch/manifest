"""Serializer."""

import json
import os
from pathlib import Path
from typing import Dict

import xxhash

from manifest.caches.array_cache import ArrayCache


class Serializer:
    """Serializer."""

    def request_to_key(self, request: Dict) -> str:
        """
        Normalize a request into a key.

        Args:
            request: request to normalize.

        Returns:
            normalized key.
        """
        return json.dumps(request, sort_keys=True)

    def key_to_request(self, key: str) -> Dict:
        """
        Convert the normalized version to the request.

        Args:
            key: normalized key to convert.

        Returns:
            unnormalized request dict.
        """
        return json.loads(key)

    def response_to_key(self, response: Dict) -> str:
        """
        Normalize a response into a key.

        Args:
            response: response to normalize.

        Returns:
            normalized key.
        """
        return json.dumps(response, sort_keys=True)

    def key_to_response(self, key: str) -> Dict:
        """
        Convert the normalized version to the response.

        Args:
            key: normalized key to convert.

        Returns:
            unnormalized response dict.
        """
        return json.loads(key)


class ArraySerializer(Serializer):
    """Serializer for array."""

    def __init__(self) -> None:
        """
        Initialize array serializer.

        We don't want to cache the array. We hash the value and
        store the array in a memmap file. Store filename/offsets
        in sqlitedict to keep track of hash -> array.
        """
        super().__init__()

        self.hash = xxhash.xxh64()
        manifest_home = Path(os.environ.get("MANIFEST_HOME", Path.home()))
        cache_folder = manifest_home / ".manifest" / "array_cache"
        self.writer = ArrayCache(cache_folder)

    def response_to_key(self, response: Dict) -> str:
        """
        Normalize a response into a key.

        Convert arrays to hash string for cache key.

        Args:
            response: response to normalize.

        Returns:
            normalized key.
        """
        # Assume response is a dict with keys "choices" -> List dicts
        # with keys "array".
        choices = response["choices"]
        # We don't want to modify the response in place
        # but we want to avoid calling deepcopy on an array
        del response["choices"]
        response_copy = response.copy()
        response["choices"] = choices
        response_copy["choices"] = []
        for choice in choices:
            if "array" not in choice:
                raise ValueError(
                    f"Choice with keys {choice.keys()} does not have array key."
                )
            arr = choice["array"]
            # Avoid copying an array
            del choice["array"]
            new_choice = choice.copy()
            choice["array"] = arr

            self.hash.update(arr)
            hash_str = self.hash.hexdigest()
            self.hash.reset()
            new_choice["array"] = hash_str
            response_copy["choices"].append(new_choice)
            if not self.writer.contains_key(hash_str):
                self.writer.put(hash_str, arr)
        return json.dumps(response_copy, sort_keys=True)

    def key_to_response(self, key: str) -> Dict:
        """
        Convert the normalized version to the response.

        Convert the hash string keys to the arrays.

        Args:
            key: normalized key to convert.

        Returns:
            unnormalized response dict.
        """
        response = json.loads(key)
        for choice in response["choices"]:
            hash_str = choice["array"]
            choice["array"] = self.writer.get(hash_str)
        return response
