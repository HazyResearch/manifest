"""Cache for queries and responses."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from manifest.caches.serializers import ArraySerializer, NumpyByteSerializer, Serializer
from manifest.response import RESPONSE_CONSTRUCTORS, Response

# Non-text return type caches
ARRAY_CACHE_TYPES = {
    "diffuser",
    "tomadiffuser",
    "openaiembedding",
    "huggingfaceembedding",
}


class Cache(ABC):
    """A cache for request/response pairs."""

    def __init__(
        self,
        connection_str: str,
        client_name: str = "None",
        cache_args: Dict[str, Any] = {},
    ):
        """
        Initialize cache.

        Args:
            connection_str: connection string.
            client_name: name of client.
            cache_args: arguments for cache.

        cache_args are any arguments needed to initialize the cache.

        Further, cache_args can contain `array_serializer` as a string
        for embedding or image return types (e.g. diffusers) with values
        as `local_file` or `byte_string`. `local_file` will save the
        array in a local file and cache a pointer to the file.
        `byte_string` will convert the array to a byte string and cache
        the entire byte string. `byte_string` is default.

        Args:
            connection_str: connection string for client.
            cache_args: cache arguments.
        """
        self.client_name = client_name
        self.connect(connection_str, cache_args)
        if self.client_name in ARRAY_CACHE_TYPES:
            array_serializer = cache_args.pop("array_serializer", "byte_string")
            if array_serializer not in ["local_file", "byte_string"]:
                raise ValueError(
                    "array_serializer must be local_file or byte_string,"
                    f" not {array_serializer}"
                )
            self.serializer = (
                ArraySerializer()
                if array_serializer == "local_file"
                else NumpyByteSerializer()
            )
        else:
            # If user has array_serializer type, it will throw an error as
            # it is not recognized for non-array return types.
            self.serializer = Serializer()

    @abstractmethod
    def close(self) -> None:
        """Close the client."""
        raise NotImplementedError()

    @abstractmethod
    def connect(self, connection_str: str, cache_args: Dict[str, Any]) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_key(self, key: str, table: str = "default") -> Union[str, None]:
        """
        Get the key for a request.

        With return None if key is not in cache.

        Args:
            key: key for cache.
            table: table to get key in.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_key(self, key: str, value: str, table: str = "default") -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: key for cache.
            value: new value for key.
            table: table to set key in.
        """
        raise NotImplementedError()

    @abstractmethod
    def commit(self) -> None:
        """Commit any results."""
        raise NotImplementedError()

    def get(self, request: Dict) -> Union[Response, None]:
        """Get the result of request (by calling compute as needed).

        Args:
            request: request to get.
            response: response to get.

        Returns:
            Response object or None if not in cache.
        """
        key = self.serializer.request_to_key(request)
        cached_response = self.get_key(key)
        if cached_response:
            cached = True
            response = self.serializer.key_to_response(cached_response)
            return Response(
                response,
                cached,
                request,
                **RESPONSE_CONSTRUCTORS.get(self.client_name, {}),
            )
        return None

    def set(self, request: Dict, response: Dict) -> None:
        """Set the value for the key.

        Args:
            request: request to set.
            response: response to set.
        """
        key = self.serializer.request_to_key(request)
        self.set_key(key, self.serializer.response_to_key(response))
