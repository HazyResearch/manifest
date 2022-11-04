"""Noop cache."""
from typing import Any, Dict, Union

from manifest.caches.cache import Cache


class NoopCache(Cache):
    """A Noop cache that caches nothing for request/response pairs."""

    def connect(self, connection_str: str, cache_args: Dict[str, Any]) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
            cache_args: cache arguments.
        """
        pass

    def close(self) -> None:
        """Close the client."""
        pass

    def get_key(self, key: str, table: str = "default") -> Union[str, None]:
        """
        Return None key for never in cache.

        Args:
            key: key for cache.
            table: table to get key in.
        """
        return None

    def set_key(self, key: str, value: str, table: str = "default") -> None:
        """
        Do not set anything as no cache.

        Args:
            key: key for cache.
            value: new value for key.
            table: table to set key in.
        """
        pass

    def commit(self) -> None:
        """Commit any results."""
        pass
