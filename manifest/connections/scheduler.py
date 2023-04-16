"""Request client schedulers.

Supports random selection and round robin selection.
"""
import numpy as np


class Scheduler:
    """Scheduler base class."""

    NAME: str = "scheduler"

    def __init__(self, num_clients: int):
        """Initialize scheduler."""
        self.num_clients = num_clients

    def get_client(self) -> int:
        """Get client by id."""
        raise NotImplementedError


class RandomScheduler(Scheduler):
    """Random scheduler."""

    NAME: str = "random"

    def __init__(self, num_clients: int):
        """Initialize scheduler."""
        super().__init__(num_clients)
        # Set seed
        np.random.seed(0)

    def get_client(self) -> int:
        """Get client by id."""
        return np.random.randint(self.num_clients)


class RoundRobinScheduler(Scheduler):
    """Round robin scheduler."""

    NAME: str = "round_robin"

    def __init__(self, num_clients: int):
        """Initialize scheduler."""
        super().__init__(num_clients)
        self.current_client = 0

    def get_client(self) -> int:
        """Get client by id."""
        client = self.current_client
        self.current_client = (self.current_client + 1) % self.num_clients
        return client
