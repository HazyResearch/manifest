"""Redis cache."""
from typing import Any, Dict, Union

import redis

from manifest.caches.cache import Cache


class RedisCache(Cache):
    """A Redis cache for request/response pairs."""

    def connect(self, connection_str: str, cache_args: Dict[str, Any]) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
            cache_args: cache arguments.
        """
        host, port = connection_str.split(":")
        self.redis = redis.Redis(host=host, port=int(port), db=0)
        return

    def close(self) -> None:
        """Close the client."""
        self.redis.close()

    def _normalize_table_key(self, key: str, table: str) -> str:
        """Cast key for prompt key."""
        return f"{table}:{key}"

    def get_key(self, key: str, table: str = "default") -> Union[str, None]:
        """
        Get the key for a request.

        With return None if key is not in cache.

        Args:
            key: key for cache.
            table: table to get key in.
        """
        norm_key = self._normalize_table_key(key, table)
        if self.redis.exists(norm_key):
            return self.redis.get(norm_key).decode("utf-8")
        else:
            return None

    def set_key(self, key: str, value: str, table: str = "default") -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: key for cache.
            value: new value for key.
            table: table to set key in.
        """
        self.redis.set(self._normalize_table_key(key, table), value)
        self.commit()

    def commit(self) -> None:
        """Commit any results."""
        pass
