"""Dummy client."""
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from manifest.clients import Client
from manifest.clients.response import Response

logger = logging.getLogger(__name__)


class DummyClient(Client):
    """Dummy client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        num_results: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:
        """
        Connect to dummpy server.

        This is a dummy client that returns identity responses. Used for testing.
        """
        self.num_results = num_results

    def close(self) -> None:
        """Close the client."""
        pass

    def get_request(
        self, query: str, **kwargs: Any
    ) -> Tuple[Callable[[], Response], Dict]:
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
            "num_results": kwargs.get("num_results", self.num_results),
        }

        def _run_completion() -> Response:
            return Response({"choices": [{"text": "hello"}] * self.num_results})

        return _run_completion, request_params
