"""AI21 client."""
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from manifest.clients.client import Client

logger = logging.getLogger(__name__)

AI21_ENGINES = {
    "j1-jumbo",
    "j1-grande",
    "j1-large",
}

# User param -> (client param, default value)
AI21_PARAMS = {
    "engine": ("engine", "j1-large"),
    "temperature": ("temperature", 1.0),
    "max_tokens": ("maxTokens", 10),
    "top_k_return": ("topKReturn", 0),
    "n": ("numResults", 1),
    "top_p": ("topP", 1.0),
    "stop_sequences": ("stopSequences", []),
    "client_timeout": ("client_timeout", 60),  # seconds
}


class AI21Client(Client):
    """AI21Client client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the AI21 server.

        connection_str is passed as default AI21_API_KEY if variable not set.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        # Taken from https://studio.ai21.com/docs/api/
        self.host = "https://api.ai21.com/studio/v1"
        self.api_key = os.environ.get("AI21_API_KEY", connection_str)
        if self.api_key is None:
            raise ValueError(
                "AI21 API key not set. Set AI21_API_KEY environment "
                "variable or pass through `connection_str`."
            )

        for key in AI21_PARAMS:
            setattr(self, key, client_args.pop(key, AI21_PARAMS[key][1]))
        if getattr(self, "engine") not in AI21_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {AI21_ENGINES}."
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
        return {"model_name": "ai21", "engine": getattr(self, "engine")}

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(AI21_PARAMS.keys())

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
                    "text": item["data"]["text"],
                    "logprobs": item["data"]["tokens"],
                }
                for item in response["completions"]
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
        for key in AI21_PARAMS:
            if key in ["client_timeout"]:
                # These are not passed to the AI21 API
                continue
            request_params[AI21_PARAMS[key][0]] = request_args.pop(
                key, getattr(self, key)
            )

        def _run_completion() -> Dict:
            post_str = self.host + "/" + getattr(self, "engine") + "/complete"
            res = requests.post(
                post_str,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_params,
            )
            try:
                res = requests.post(
                    post_str,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=request_params,
                    timeout=getattr(self, "client_timeout"),
                )
                res.raise_for_status()
            except requests.Timeout as e:
                logger.error("AI21 request timed out. Increase client_timeout.")
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
        raise NotImplementedError("AI21 does not support choice logit request.")
