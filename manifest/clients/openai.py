"""OpenAI client."""
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from manifest.clients.client import Client

logger = logging.getLogger(__name__)

OPENAI_ENGINES = {
    "text-davinci-002",
    "text-davinci-001",
    "davinci",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
    "code-davinci-002",
    "code-cushman-001",
}

# User param -> (client param, default value)
OPENAI_PARAMS = {
    "engine": ("model", "text-davinci-002"),
    "temperature": ("temperature", 1.0),
    "max_tokens": ("max_tokens", 10),
    "n": ("n", 1),
    "top_p": ("top_p", 1.0),
    "logprobs": ("logprobs", None),
    "top_k_return": ("best_of", 1),
    "stop_sequence": ("stop", None),  # OpenAI doesn't like empty lists
    "presence_penalty": ("presence_penalty", 0.0),
    "frequency_penalty": ("frequency_penalty", 0.0),
    "client_timeout": ("client_timeout", 60),  # seconds
}


class OpenAIClient(Client):
    """OpenAI client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the OpenAI server.

        connection_str is passed as default OPENAI_API_KEY if variable not set.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.api_key = os.environ.get("OPENAI_API_KEY", connection_str)
        if self.api_key is None:
            raise ValueError(
                "OpenAI API key not set. Set OPENAI_API_KEY environment "
                "variable or pass through `connection_str`."
            )
        self.host = "https://api.openai.com/v1"
        for key in OPENAI_PARAMS:
            setattr(self, key, client_args.pop(key, OPENAI_PARAMS[key][1]))
        if getattr(self, "engine") not in OPENAI_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {OPENAI_ENGINES}."
            )

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
        return {"model_name": "openai", "engine": getattr(self, "engine")}

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(OPENAI_PARAMS.keys())

    def format_response(self, response: Dict) -> Dict[str, Any]:
        """
        Format response to dict.

        Args:
            response: response

        Return:
            response as dict
        """
        if "choices" not in response:
            raise ValueError(f"Invalid response: {response}")
        return response

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
        for key in OPENAI_PARAMS:
            if key in ["client_timeout"]:
                # These are not passed to the OpenAI API
                continue
            request_params[OPENAI_PARAMS[key][0]] = request_args.pop(
                key, getattr(self, key)
            )

        def _run_completion() -> Dict:
            post_str = self.host + "/completions"
            try:
                res = requests.post(
                    post_str,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=request_params,
                    timeout=getattr(self, "client_timeout"),
                )
                res.raise_for_status()
            except requests.Timeout as e:
                logger.error("OpenAI request timed out. Increase client_timeout.")
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
        raise NotImplementedError("OpenAI does not support choice logit request.")
