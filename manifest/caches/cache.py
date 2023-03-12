"""Cache for queries and responses."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from manifest.caches.serializers import ArraySerializer, Serializer
from manifest.response import RESPONSE_CONSTRUCTORS, Response

CACHE_CONSTRUCTOR = {
    "diffuser": ArraySerializer,
    "tomadiffuser": ArraySerializer,
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
        Initialize client.

        Args:
            connection_str: connection string.
            client_name: name of client.
            cache_args: arguments for cache.

        cache_args are passed to client as default parameters.

        For clients like OpenAI that do not require a connection,
        the connection_str can be None.

        Args:
            connection_str: connection string for client.
            cache_args: cache arguments.
        """
        self.client_name = client_name
        self.connect(connection_str, cache_args)
        self.serializer = CACHE_CONSTRUCTOR.get(client_name, Serializer)()

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
                **RESPONSE_CONSTRUCTORS.get(self.client_name, {})
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
