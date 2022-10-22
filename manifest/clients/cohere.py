"""Cohere client."""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cohere

from manifest.clients.client import Client

logging.getLogger("cohere").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

COHERE_MODELS = {"small", "medium", "large", "xlarge"}

# Params are defined in https://docs.cohere.ai/generate-reference
COHERE_PARAMS = {
    "model": ("model", "xlarge"),
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
        api_key = os.environ.get("COHERE_API_KEY", connection_str)
        if api_key is None:
            raise ValueError(
                "Cohere API key not set. Set COHERE_API_KEY environment "
                "variable or pass through `connection_str`."
            )
        self.co = cohere.Client(api_key)
        for key in COHERE_PARAMS:
            setattr(self, key, client_args.pop(key, COHERE_PARAMS[key][1]))
        if getattr(self, "model") not in COHERE_MODELS:
            raise ValueError(
                f"Invalid model {getattr(self, 'model')}. Must be {COHERE_MODELS}."
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
        return {"model_name": "model", "model": getattr(self, "model")}

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(COHERE_PARAMS.keys())

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
            request_params[COHERE_PARAMS[key][0]] = request_args.pop(
                key, getattr(self, key)
            )

        def _run_generation() -> Dict:
            try:
                response = self.co.generate(**request_params)
                return json.loads(
                    json.dumps(
                        response, default=lambda o: getattr(o, "__dict__", str(o))
                    )
                )
            except cohere.CohereError as e:
                logger.error(e)
                raise e

        return _run_generation, request_params

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
