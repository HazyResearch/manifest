"""Test session."""
import sqlite3
from pathlib import Path

import pytest

from manifest.session import Session


@pytest.mark.usefixtures("session_cache")
def test_init(session_cache: Path) -> None:
    """Test session initialization."""
    session = Session()
    assert isinstance(session.conn, sqlite3.Connection)
    assert session.db_file == session_cache / ".manifest" / "session.db"
    assert session.query_id == 0
    assert (session_cache / ".manifest" / "session.db").exists()
    # Remove session cache file.
    (session_cache / ".manifest" / "session.db").unlink()

    session = Session("dog_days")
    assert isinstance(session.conn, sqlite3.Connection)
    assert session.db_file == session_cache / ".manifest" / "session.db"
    assert session.query_id == 0
    assert session.session_id == "dog_days"
    assert (session_cache / ".manifest" / "session.db").exists()
    session.close()


@pytest.mark.usefixtures("session_cache")
def test_log_query(session_cache: Path) -> None:
    """Test session log_query."""
    session = Session()
    assert session.get_last_queries(1) == []

    query_key = {"query": "What is your name?", "time": "now"}
    response_key = {"response": "I don't have a name", "engine": "nodel"}
    session.log_query(query_key, response_key)
    assert session.query_id == 1
    assert session.get_last_queries(1) == [(query_key, response_key)]

    query_key2 = {"query2": "What is your name?", "time": "now"}
    response_key2 = {"response2": "I don't have a name", "engine": "nodel"}
    session.log_query(query_key2, response_key2)
    assert session.query_id == 2
    assert len(session.get_last_queries(1)) == 1
    assert session.get_last_queries(2) == [
        (query_key, response_key),
        (query_key2, response_key2),
    ]
    session.close()


@pytest.mark.usefixtures("session_cache")
def test_resume_query(session_cache: Path) -> None:
    """Test session log_query."""
    session = Session(session_id="dog_days")
    query_key = {"query": "What is your name?", "time": "now"}
    response_key = {"response": "I don't have a name", "engine": "nodel"}
    session.log_query(query_key, response_key)
    session.close()

    session = Session(session_id="dog_days")
    assert session.query_id == 1


@pytest.mark.usefixtures("session_cache")
def test_session_keys(session_cache: Path) -> None:
    """Test get session keys."""
    # Assert empty before queries
    assert Session.get_session_keys(session_cache / ".manifest" / "session.db") == []
    # Add queries and make sure session is logged
    session = Session(session_id="dog_days")
    query_key = {"query": "What is your name?", "time": "now"}
    response_key = {"response": "I don't have a name", "engine": "nodel"}
    session.log_query(query_key, response_key)
    session.close()

    assert Session.get_session_keys(session_cache / ".manifest" / "session.db") == [
        "dog_days"
    ]
