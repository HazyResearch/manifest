"""Client class."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from manifest.clients.response import Response


class Client(ABC):
    """Client class."""

    def __init__(self, connection_str: Optional[str] = None, **kwargs: Any):
        """
        Initialize client.

        kwargs are passed to client as default parameters.

        For clients like OpenAI that do not require a connection,
        the connection_str can be None.

        Args:
            connection_str: connection string for client.
        """
        self.connect(connection_str, **kwargs)

    @abstractmethod
    def close(self) -> None:
        """Close the client."""
        raise NotImplementedError()

    @abstractmethod
    def connect(self, connection_str: str, **kwargs: Any) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_request(
        self, query: str, **kwargs: Any
    ) -> Tuple[Callable[[], Response], Dict]:
        """
        Get request function.

        kwargs override default parameters.

        Calling the returned function will run the request.

        Args:
            query: query string.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        raise NotImplementedError()
