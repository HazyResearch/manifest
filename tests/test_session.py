"""Test session."""
import sqlite3

import pytest

from manifest.session import Session


@pytest.mark.usefixtures("session_cache")
def test_init(session_cache):
    """Test session initialization."""
    session = Session()
    assert isinstance(session.conn, sqlite3.Connection)
    assert session.query_id == 0
    assert (session_cache / ".manifest" / "session.db").exists()
    # Remove session cache file.
    (session_cache / ".manifest" / "session.db").unlink()

    session = Session("dog_days")
    assert isinstance(session.conn, sqlite3.Connection)
    assert session.query_id == 0
    assert session.session_id == "dog_days"
    assert (session_cache / ".manifest" / "session.db").exists()


@pytest.mark.usefixtures("session_cache")
def test_log_query(session_cache):
    """Test session log_query."""
    session = Session()
    assert session.get_last_queries(1, return_raw_values=True) == []

    query_key = {"query": "What is your name?", "time": "now"}
    response_key = {"response": "I don't have a name", "engine": "nodel"}
    session.log_query(query_key, response_key)
    assert session.query_id == 1
    assert session.get_last_queries(1, return_raw_values=True) == [
        (query_key, response_key)
    ]

    query_key2 = {"query2": "What is your name?", "time": "now"}
    response_key2 = {"response2": "I don't have a name", "engine": "nodel"}
    session.log_query(query_key2, response_key2)
    assert session.query_id == 2
    assert session.get_last_queries(2, return_raw_values=True) == [
        (query_key, response_key),
        (query_key2, response_key2),
    ]


@pytest.mark.usefixtures("session_cache")
def test_resume_query(session_cache):
    """Test session log_query."""
    session = Session(session_id="dog_days")
    query_key = {"query": "What is your name?", "time": "now"}
    response_key = {"response": "I don't have a name", "engine": "nodel"}
    session.log_query(query_key, response_key)
    session.close()

    session = Session(session_id="dog_days")
    assert session.query_id == 1