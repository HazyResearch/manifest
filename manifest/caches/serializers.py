import json
import os
from pathlib import Path
from typing import Dict

import xxhash
from sqlitedict import SqliteDict


class Serializer:
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
    def __init__(self):
        """
        Initialize array serializer.

        We don't want to cache the array. We hash the value and
        store the array in a memmap file. Store filename/offsets
        in sqlitedict to keep track of hash -> array.
        """
        super().__init__()

        self.hash = xxhash.xxh64()
        manifest_home = Path(os.environ.get("MANIFEST_HOME", Path.home()))
        self.cache_folder = manifest_home / ".manifest" / "array_cache"
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.hash2file = SqliteDict(
            self.cache_folder / "hash2file.sqlite", autocommit=True
        )

    def response_to_key(self, response: Dict) -> str:
        """
        Normalize a response into a key.

        Args:
            response: response to normalize.

        Returns:
            normalized key.
        """
        # Assume response is a dict with keys "choices" -> List dicts
        # with keys "array".
        for choice in response["choices"]:
            arr = choice["array"]
            self.hash.update(arr)
            hash_str = self.hash.hexdigest()
            self.hash.reset()
            choice["array"] = hash_str
            # TODO: implement memmap saving here
            if hash_str not in self.hash2file:
                file_name = self.cache_folder / hash_str
                with open(file_name, "wb") as f:
                    f.write(arr)
                self.hash2file[hash_str] = file_name
        return json.dumps(response, sort_keys=True)

    def key_to_response(self, key: str) -> Dict:
        """
        Convert the normalized version to the response.

        Args:
            key: normalized key to convert.

        Returns:
            unnormalized response dict.
        """
        # TODO: support diffusers
        res = json.loads(key)
        raise NotImplementedError()
