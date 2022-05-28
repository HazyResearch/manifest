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
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the HuggingFace url.

        Arsg:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.host = connection_str.rstrip("/")
        self.temperature = client_args.pop("temperature", 0.00001)
        self.max_tokens = client_args.pop("max_tokens", 10)
        self.top_p = client_args.pop("top_p", 1.0)
        self.top_k = client_args.pop("top_k", 50)
        self.repetition_penalty = client_args.pop("repetition_penalty", 1.0)
        self.n = client_args.pop("n", 1)
        self.do_sample = client_args.pop("do_sample", True)
        self.model_params = self.get_model_params()

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
        res = requests.post(self.host + "/params")
        return res.json()

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
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "repetition_penalty": kwargs.get(
                "repetition_penalty", self.repetition_penalty
            ),
            "n": kwargs.get("n", self.n),
        }
        request_params.update(self.model_params)

        def _run_completion() -> Dict:
            post_str = self.host + "/completions"
            res = requests.post(post_str, json=request_params)
            return res.json()

        return _run_completion, request_params
