"""SQLite cache."""
import logging
from typing import Any, Dict, Union

from sqlitedict import SqliteDict

from manifest.caches.cache import Cache

logging.getLogger("sqlitedict").setLevel(logging.WARNING)


class SQLiteCache(Cache):
    """A SQLite cache for request/response pairs."""

    def connect(self, connection_str: str, cache_args: Dict[str, Any]) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
            cache_args: cache arguments.
        """
        self.cache_file = connection_str
        if not self.cache_file:
            self.cache_file = ".sqlite.cache"
        self.cache = SqliteDict(self.cache_file, autocommit=True)
        return

    def close(self) -> None:
        """Close the client."""
        self.cache.close()

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
        return self.cache.get(self._normalize_table_key(key, table))

    def set_key(self, key: str, value: str, table: str = "default") -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: key for cache.
            value: new value for key.
            table: table to set key in.
        """
        self.cache[self._normalize_table_key(key, table)] = value
        self.commit()

    def commit(self) -> None:
        """Commit any results."""
        self.cache.commit()
