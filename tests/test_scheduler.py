"""Test scheduler."""

from manifest.connections.scheduler import RandomScheduler, RoundRobinScheduler


def test_random_scheduler() -> None:
    """Test random scheduler."""
    scheduler = RandomScheduler(num_clients=2)
    # Try 20 clients and make sure 0 and 1 are both
    # returned
    client_ids = set()
    for _ in range(20):
        client_id = scheduler.get_client()
        assert client_id in [0, 1]
        client_ids.add(client_id)
    assert len(client_ids) == 2


def test_round_robin_scheduler() -> None:
    """Test round robin scheduler."""
    scheduler = RoundRobinScheduler(num_clients=2)
    assert scheduler.get_client() == 0
    assert scheduler.get_client() == 1
    assert scheduler.get_client() == 0
    assert scheduler.get_client() == 1
