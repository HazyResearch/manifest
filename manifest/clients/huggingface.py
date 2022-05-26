"""OpenAI client."""
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import requests

from manifest.clients.client import Client

logger = logging.getLogger(__name__)


class HuggingFaceClient(Client):
    """HuggingFace client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        max_tokens: Optional[int] = 10,
        top_p: Optional[float] = 1.0,
        top_k: Optional[int] = 0,
        repetition_penalty: Optional[float] = 1.0,
        n: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:
        """Connect to the HuggingFace url."""
        self.host = connection_str.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.n = n

    def close(self) -> None:
        """Close the client."""
        pass

    def get_request(self, query: str, **kwargs: Any) -> Tuple[Callable[[], Dict], Dict]:
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
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "repetition_penalty": kwargs.get(
                "repetition_penalty", self.repetition_penalty
            ),
            "n": kwargs.get("n", self.n),
        }

        def _run_completion() -> Dict:
            post_str = self.host + "/completions"
            res = requests.post(post_str, json=request_params)
            return res.json()

        return _run_completion, request_params
