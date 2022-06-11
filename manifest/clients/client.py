"""Client class."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple


class Client(ABC):
    """Client class."""

    def __init__(
        self, connection_str: Optional[str] = None, client_args: Dict[str, Any] = {}
    ):
        """
        Initialize client.

        kwargs are passed to client as default parameters.

        For clients like OpenAI that do not require a connection,
        the connection_str can be None.

        Args:
            connection_str: connection string for client.
            client_args: client arguments.
        """
        self.connect(connection_str, client_args)

    @abstractmethod
    def close(self) -> None:
        """Close the client."""
        raise NotImplementedError()

    @abstractmethod
    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        raise NotImplementedError()

    @abstractmethod
    def connect(self, connection_str: str, client_args: Dict[str, Any]) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_request(
        self, query: str, request_args: Dict[str, Any] = {}
    ) -> Tuple[Callable[[], Dict], Dict]:
        """
        Get request function.

        kwargs override default parameters.

        Calling the returned function will run the request.

        Args:
            query: query string.
            request_args: request arguments.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        raise NotImplementedError()
