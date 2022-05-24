"""SQLite cache."""
import logging
from pathlib import Path
from typing import Any, Union

from sqlitedict import SqliteDict

from manifest.caches import Cache

logging.getLogger("sqlitedict").setLevel(logging.WARNING)


class SQLiteCache(Cache):
    """A SQLite cache for request/response pairs."""

    def connect(self, connection_str: str, **kwargs: Any) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
        """
        self.cache_dir = connection_str
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        # If more than two tables, switch to full on SQL connection
        self.query_file = Path(self.cache_dir, "query.sqlite")
        self.prompt_file = Path(self.cache_dir, "prompts.sqlite")
        self.cache = SqliteDict(self.query_file, autocommit=False)
        self.prompt_cache = SqliteDict(self.prompt_file, autocommit=False)
        return

    def close(self) -> None:
        """Close the client."""
        self.cache.close()

    def get_key(self, key: str, table: str = "default") -> Union[str, None]:
        """
        Get the key for a request.

        With return None if key is not in cache.

        Args:
            key: key for cache.
            table: table to get key in.
        """
        if table == "prompt":
            return self.prompt_cache.get(key)
        else:
            if table != "default":
                raise ValueError(
                    "SQLiteDict only support table of `default` or `prompt`"
                )
        return self.cache.get(key)

    def set_key(self, key: str, value: str, table: str = "default") -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: key for cache.
            value: new value for key.
            table: table to set key in.
        """
        if table == "prompt":
            self.prompt_cache[key] = value
        else:
            if table != "default":
                raise ValueError(
                    "SQLiteDict only support table of `default` or `prompt`"
                )
            self.cache[key] = value
        self.commit()

    def commit(self) -> None:
        """Commit any results."""
        self.prompt_cache.commit()
        self.cache.commit()
