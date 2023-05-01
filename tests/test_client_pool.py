"""Test client pool."""

import time

import pytest

from manifest.connections.client_pool import ClientConnection, ClientConnectionPool
from manifest.request import LMRequest


def test_init() -> None:
    """Test initialization."""
    client_connection1 = ClientConnection(
        client_name="openai", client_connection="XXX", engine="text-davinci-002"
    )
    client_connection2 = ClientConnection(
        client_name="openai", client_connection="XXX", engine="text-ada-001"
    )
    client_connection3 = ClientConnection(
        client_name="openaiembedding", client_connection="XXX"
    )
    with pytest.raises(ValueError) as exc_info:
        ClientConnectionPool(
            [client_connection1, client_connection2], client_pool_scheduler="bad"
        )
    assert str(exc_info.value) == "Unknown scheduler: bad."

    with pytest.raises(ValueError) as exc_info:
        ClientConnectionPool([client_connection1, client_connection3])
    assert (
        str(exc_info.value)
        == "All clients in the client pool must use the same request type. You have [\"<class 'manifest.request.EmbeddingRequest'>\", \"<class 'manifest.request.LMRequest'>\"]"  # noqa: E501"
    )

    pool = ClientConnectionPool([client_connection1, client_connection2])
    assert pool.request_type == LMRequest
    assert len(pool.client_pool) == 2
    assert len(pool.client_pool_metrics) == 2
    assert pool.client_pool[0].engine == "text-davinci-002"  # type: ignore
    assert pool.client_pool[1].engine == "text-ada-001"  # type: ignore


def test_timing() -> None:
    """Test timing client."""
    client_connection1 = ClientConnection(client_name="dummy")
    client_connection2 = ClientConnection(client_name="dummy")
    connection_pool = ClientConnectionPool([client_connection1, client_connection2])

    connection_pool.get_next_client()
    assert connection_pool.current_client_id == 0
    connection_pool.start_timer()
    time.sleep(2)
    connection_pool.end_timer()

    connection_pool.get_next_client()
    assert connection_pool.current_client_id == 1
    connection_pool.start_timer()
    time.sleep(1)
    connection_pool.end_timer()

    timing = connection_pool.client_pool_metrics
    assert timing[0].end - timing[0].start > 1.9
    assert timing[1].end - timing[1].start > 0.9
