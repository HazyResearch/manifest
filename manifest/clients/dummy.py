"""Dummy client."""
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from manifest.clients import Client

logger = logging.getLogger(__name__)


class DummyClient(Client):
    """Dummy client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to dummpy server.

        This is a dummy client that returns identity responses. Used for testing.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.num_results = client_args.pop("num_results", 1)

    def close(self) -> None:
        """Close the client."""
        pass

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        return {"engine": "dummy"}

    def get_request(
        self, query: str, request_args: Dict[str, Any] = {}
    ) -> Tuple[Callable[[], Dict], Dict]:
        """
        Get request string function.

        Args:
            query: query string.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        request_params = {
            "prompt": query,
            "num_results": request_args.pop("num_results", self.num_results),
        }

        def _run_completion() -> Dict:
            return {"choices": [{"text": "hello"}] * request_params["num_results"]}

        return _run_completion, request_params
