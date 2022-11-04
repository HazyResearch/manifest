"""Cohere client."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from manifest.clients.client import Client

logger = logging.getLogger(__name__)

COHERE_MODELS = {"small", "medium", "large", "xlarge"}

# Params are defined in https://docs.cohere.ai/generate-reference
COHERE_PARAMS = {
    "engine": ("model", "xlarge"),
    "max_tokens": ("max_tokens", 20),
    "temperature": ("temperature", 0.75),
    "num_generations": ("num_generations", 1),
    "k": ("k", 0),
    "p": ("p", 0.75),
    "frequency_penalty": ("frequency_penalty", 0.0),
    "presence_penalty": ("presence_penalty", 0.0),
    "stop_sequences": ("stop_sequences", []),
    "return_likelihoods": ("return_likelihoods", ""),
    "logit_bias": ("logit_bias", {}),
    "client_timeout": ("client_timeout", 60),  # seconds
}


class CohereClient(Client):
    """Cohere client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the Cohere server.

        connection_str is passed as default COHERE_API_KEY if variable not set.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.api_key = os.environ.get("COHERE_API_KEY", connection_str)
        if self.api_key is None:
            raise ValueError(
                "Cohere API key not set. Set COHERE_API_KEY environment "
                "variable or pass through `connection_str`."
            )
        self.host = "https://api.cohere.ai"
        for key in COHERE_PARAMS:
            setattr(self, key, client_args.pop(key, COHERE_PARAMS[key][1]))
        if getattr(self, "engine") not in COHERE_MODELS:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {COHERE_MODELS}."
            )

    def close(self) -> None:
        """Close the client."""

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        return {"model_name": "cohere", "engine": getattr(self, "engine")}

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(COHERE_PARAMS.keys())

    def format_response(self, response: Dict) -> Dict[str, Any]:
        """
        Format response to dict.

        Args:
            response: response

        Return:
            response as dict
        """
        return {
            "object": "text_completion",
            "model": getattr(self, "engine"),
            "choices": [
                {
                    "text": item["text"],
                    "text_logprob": item.get("likelihood", None),
                    "logprobs": item.get("token_likelihoods", None),
                }
                for item in response["generations"]
            ],
        }

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
        for key in COHERE_PARAMS:
            if key in ["client_timeout"]:
                continue
            request_params[COHERE_PARAMS[key][0]] = request_args.pop(
                key, getattr(self, key)
            )

        def _run_completion() -> Dict:
            post_str = self.host + "/generate"
            try:
                res = requests.post(
                    post_str,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Cohere-Version": "2021-11-08",
                    },
                    json=request_params,
                    timeout=getattr(self, "client_timeout"),
                )
                res.raise_for_status()
            except requests.Timeout as e:
                logger.error("Cohere request timed out. Increase client_timeout.")
                raise e
            except requests.exceptions.HTTPError as e:
                raise e
            return self.format_response(res.json())

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
        raise NotImplementedError("Cohere does not support choice logit request.")
