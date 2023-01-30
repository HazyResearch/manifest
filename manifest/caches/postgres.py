"""SQLite cache."""
import logging
from typing import Any, Dict, Union

import pg8000
import sqlalchemy
from sqlalchemy.dialects.postgresql import HSTORE
from sqlalchemy import Column, Integer, String, Boolean, DateTime


from manifest.caches.cache import Cache

logging.getLogger("sqlitedict").setLevel(logging.WARNING)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

Base = declarative_base()


class Request(Base):
    __tablename__ = "requests"
    key = Column(String, primary_key=True)
    response = Column(String)   # FIXME: this should be an hstore ideally, but I don't want to set it up on GCP


class PostgreSQLCache(Cache):
    """A PostgreSQL cache for request/response pairs."""

    def connect(
        self, 
        connection_str: str, 
        cache_args: Dict[str, Any]
    ) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
            cache_args: arguments for cache.
        """
        from google.cloud.sql.connector import Connector
        connector = Connector()

        def getconn():
            conn = connector.connect(
                connection_str,
                "pg8000",
                user=cache_args.pop("user"),
                password=cache_args.pop("password"),
                db=cache_args.pop("db"),
            )
            return conn

        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
        )
        engine.dialect.description_encoding = None

        db_exists = len(sqlalchemy.inspect(engine).get_table_names()) > 0
        if not db_exists:
            print("Creating database...")
            Base.metadata.create_all(engine)

        self.session = sessionmaker(bind=engine)()

    def close(self) -> None:
        """Close the client."""
        self.session.close() 

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
        request = self.session.query(Request).filter_by(key=key).first()
        out = request.response if request else None
        return out

    def set_key(self, key: str, value: str, table: str = "default") -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: key for cache.
            value: new value for key.
            table: table to set key in.
        """
        request = self.session.query(Request).filter_by(key=key).first()
        if request:
            request.response = value
        else:
            self.session.add(Request(key=key, response=value))
        self.commit()

    def commit(self) -> None:
        """Commit any results."""
        self.session.commit()