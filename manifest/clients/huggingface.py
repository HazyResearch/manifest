"""OpenAI client."""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from manifest.clients.client import Client

logger = logging.getLogger(__name__)

# User param -> (client param, default value)
HF_PARAMS = {
    "temperature": ("temperature", 1.0),
    "max_tokens": ("max_tokens", 10),
    "n": ("n", 1),
    "top_p": ("top_p", 1.0),
    "top_k": ("top_k", 50),
    "repetition_penalty": ("repetition_penalty", 1.0),
    "do_sample": ("do_sample", True),
}


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
        for key in HF_PARAMS:
            setattr(self, key, client_args.pop(key, HF_PARAMS[key][1]))
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

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(HF_PARAMS.keys())

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
        request_params = {"prompt": query}
        for key in HF_PARAMS:
            request_params[HF_PARAMS[key][0]] = request_args.pop(
                key, getattr(self, key)
            )
        request_params.update(self.model_params)

        def _run_completion() -> Dict:
            post_str = self.host + "/completions"
            res = requests.post(post_str, json=request_params)
            return res.json()

        return _run_completion, request_params
