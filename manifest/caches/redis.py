"""Redis cache."""
from typing import Any, Union

import redis

from manifest.caches import Cache


class RedisCache(Cache):
    """A Redis cache for request/response pairs."""

    def connect(self, connection_str: str, **kwargs: Any) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
        """
        host, port = connection_str.split(":")
        self.redis = redis.Redis(host=host, port=int(port))
        return

    def close(self) -> None:
        """Close the client."""
        self.redis.close()

    def get_key(self, key: str, table: str = "default") -> Union[str, None]:
        """
        Get the key for a request.

        With return None if key is not in cache.

        Args:
            key: key for cache.
        """
        pass

    def set_key(self, key: str, value: str, table: str = "default") -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: key for cache.
            value: new value for key.
        """
        self.redis[key] = value

    def commit(self) -> None:
        """Commit any results."""
        pass
