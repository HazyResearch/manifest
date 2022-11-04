"""User query session logging."""
import logging
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from manifest.caches.cache import (
    key_to_request,
    key_to_response,
    request_to_key,
    response_to_key,
)

logging.getLogger("sqlitedict").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class Session:
    """A user session for caching requests."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        """
        Initialize session.

        If session_id already exists, will append to existing session.

        Args:
            session_id: session id.

        """
        manifest_home = Path(os.environ.get("MANIFEST_SESSION_HOME", Path.home()))
        self.db_file = manifest_home / ".manifest" / "session.db"
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_file))
        self._create_table()
        if not session_id:
            self.session_id = str(uuid.uuid4())
            self.query_id = 0
        else:
            self.session_id = session_id
            self.query_id = self._get_latest_query_id(self.session_id)
            self.query_id += 1
        logger.info(f"Starting session {self.session_id}")
        return

    def close(self) -> None:
        """Close the client."""
        self.conn.close()

    @classmethod
    def get_session_keys(cls, db_file: Path) -> List[str]:
        """Get available session keys from cached file."""
        try:
            conn = sqlite3.connect(str(db_file))
            query = """SELECT DISTINCT session_id FROM queries"""
            cur = conn.cursor()
            res = cur.execute(query)
            return [x[0] for x in res.fetchall()]
        except sqlite3.OperationalError:
            logger.info(
                "There is no database with the 'queries' table. "
                "Are you sure you are using the right session file"
            )
            return []

    def _execute_query(self, query: str, *args: Any) -> Any:
        """
        Execute query with optional args.

        Args:
            query: query to execute.
        """
        cur = self.conn.cursor()
        res = cur.execute(query, args)
        self.conn.commit()
        return res

    def _create_table(self) -> None:
        """Create table if not exists."""
        query = """CREATE TABLE IF NOT EXISTS queries (
            query_id integer NOT NULL,
            session_id text NOT NULL,
            query_key text NOT NULL,
            response_key text NOT NULL
        );"""
        self._execute_query(query)
        return

    def _get_latest_query_id(self, session_id: str) -> int:
        """
        Get latest query id issued if resuming session.

        If no session_id, return -1.

        Args:
            session_id: session id.

        Returns:
            latest query id.
        """
        query = """SELECT query_id
                    FROM queries
                    WHERE session_id = ?
                    ORDER BY query_id DESC LIMIT 1"""
        res = self._execute_query(query, session_id).fetchone()
        if res:
            return res[0]
        return -1

    def log_query(
        self, query_key: Dict[str, Any], response_key: Dict[str, Any]
    ) -> None:
        """
        Log the query and response.

        Args:
            query_key: query of user (dump of request params).
            response_key: response of server (dump of response).
        """
        query = """INSERT INTO queries VALUES (?, ?, ?, ?);"""
        self._execute_query(
            query,
            self.query_id,
            self.session_id,
            request_to_key(query_key),
            response_to_key(response_key),
        )
        self.query_id += 1
        return

    def get_last_queries(
        self, last_n: int = -1
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Get last n queries from current session.

        If last_n is -1, return all queries.

        Args:
            last_n: last n queries.

        Returns:
            last n list of queries and outputs.
        """
        first_query = self.query_id - last_n if last_n > 0 else -1
        query = """SELECT query_key, response_key
                    FROM queries
                    WHERE session_id = ? AND query_id >= ?
                    ORDER BY query_id;"""
        res = self._execute_query(query, self.session_id, first_query)
        parsed_res = [
            (key_to_request(pair[0]), key_to_response(pair[1]))
            for pair in res.fetchall()
        ]
        return parsed_res
