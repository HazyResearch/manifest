"""OPT client."""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from manifest.clients.client import Client

logger = logging.getLogger(__name__)

# User param -> (client param, default value)
OPT_PARAMS = {
    "temperature": ("temperature", 1.0),
    "max_tokens": ("max_tokens", 10),
    "n": ("n", 1),
    "top_p": ("top_p", 1.0),
}


class OPTClient(Client):
    """OPT client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the OPT url.

        Arsg:
            connection_str: connection string.
            client_args: client arguments.
        """
        if not connection_str:
            raise ValueError("Must provide connection string")
        self.host = connection_str.rstrip("/")
        for key in OPT_PARAMS:
            setattr(self, key, client_args.pop(key, OPT_PARAMS[key][1]))

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
        return {"model_name": "opt", "engine": "opt-175b"}

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(OPT_PARAMS.keys())

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
        for key in OPT_PARAMS:
            request_params[OPT_PARAMS[key][0]] = request_args.pop(
                key, getattr(self, key)
            )

        def _run_completion() -> Dict:
            post_str = self.host + "/completions"
            res = requests.post(post_str, json=request_params)
            return res.json()

        return _run_completion, request_params

    def get_choice_logit_request(
        self, query: str, gold_choices: List[str], request_args: Dict[str, Any] = {}
    ) -> Tuple[Callable[[], Dict], Dict]:
        """
        Get request string function for choosing max choices.

        Args:
            query: query string.
            gold_choices: choices for model to choose from via max logits.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        raise NotImplementedError("OPT does not support choice logit request.")
