"""Postgres cache."""
import hashlib
import logging
from typing import Any, Dict, Union

logger = logging.getLogger("postgresql")
logger.setLevel(logging.WARNING)

from ..caches.cache import Cache

try:
    import sqlalchemy  # type: ignore
    from google.cloud.sql.connector import Connector  # type: ignore
    from sqlalchemy import Column, String  # type: ignore
    from sqlalchemy.ext.declarative import declarative_base  # type: ignore
    from sqlalchemy.orm import sessionmaker  # type: ignore

    Base = declarative_base()

    class Request(Base):  # type: ignore
        """The request table."""

        __tablename__ = "requests"
        key = Column(String, primary_key=True)
        response = Column(
            String
        )  # FIXME: ideally should be an hstore, but I don't want to set it up on GCP

    missing_dependencies = None

except ImportError as e:
    missing_dependencies = e


class PostgresCache(Cache):
    """A PostgreSQL cache for request/response pairs."""

    def connect(self, connection_str: str, cache_args: Dict[str, Any]) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
            cache_args: arguments for cache should include the following fields:
                {
                    "cache_user": "",
                    "cache_password": "",
                    "cache_db": ""
                }
        """
        if missing_dependencies:
            raise ValueError(
                "Missing dependencies for GCP PostgreSQL cache. "
                "Install with `pip install manifest[gcp]`",
                missing_dependencies,
            )

        connector = Connector()

        def getconn() -> Any:
            conn = connector.connect(
                connection_str,
                "pg8000",
                user=cache_args.pop("cache_user"),
                password=cache_args.pop("cache_password"),
                db=cache_args.pop("cache_db"),
            )
            return conn

        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
        )
        engine.dialect.description_encoding = None  # type: ignore

        db_exists = len(sqlalchemy.inspect(engine).get_table_names()) > 0
        if not db_exists:
            logger.info("Creating database...")
            Base.metadata.create_all(engine)

        self.session = sessionmaker(bind=engine)()

    def close(self) -> None:
        """Close the client."""
        self.session.close()

    @staticmethod
    def _hash_key(key: str, table: str) -> str:
        """Compute MD5 hash of the key."""
        return hashlib.md5(f"{key}:{table}".encode("utf-8")).hexdigest()

    def get_key(self, key: str, table: str = "default") -> Union[str, None]:
        """
        Get the key for a request.

        With return None if key is not in cache.

        Args:
            key: key for cache.
            table: table to get key in.
        """
        request = (
            self.session.query(Request)  # type: ignore
            .filter_by(key=self._hash_key(key, table))
            .first()
        )
        out = request.response if request else None
        return out  # type: ignore

    def set_key(self, key: str, value: str, table: str = "default") -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: key for cache.
            value: new value for key.
            table: table to set key in.
        """
        key = self._hash_key(key, table)
        request = self.session.query(Request).filter_by(key=key).first()  # type: ignore
        if request:
            request.response = value  # type: ignore
        else:
            self.session.add(Request(key=key, response=value))
        self.commit()

    def commit(self) -> None:
        """Commit any results."""
        self.session.commit()
