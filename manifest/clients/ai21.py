"""OpenAI client."""
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import requests

from manifest.clients.client import Client

logger = logging.getLogger(__name__)

AI21_ENGINES = {
    "j1-jumbo",
    "j1-grande",
    "j1-large",
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
        self.engine = client_args.pop("engine", "j1-large")
        if self.engine not in AI21_ENGINES:
            raise ValueError(f"Invalid engine {self.engine}. Must be {AI21_ENGINES}.")
        self.temperature = client_args.pop("temperature", 0.0)
        self.max_tokens = client_args.pop("max_tokens", 10)
        self.top_k_return = client_args.pop("topKReturn", 1.0)
        self.num_results = client_args.pop("numResults", 1)
        self.top_p = client_args.pop("topP", 1.0)

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
        return {"model_name": "ai21", "engine": self.engine}

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
            "model": self.engine,
            "choices": [
                {
                    "text": item["data"]["text"],
                    "logprobs": [
                        {
                            "token": tok["generatedToken"]["token"],
                            "logprob": tok["generatedToken"]["logprob"],
                            "start": tok["textRange"]["start"],
                            "end": tok["textRange"]["end"],
                        }
                        for tok in item["data"]["tokens"]
                    ],
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
        request_params = {
            "engine": request_args.pop("engine", self.engine),
            "prompt": query,
            "temperature": request_args.pop("temperature", self.temperature),
            "maxTokens": request_args.pop("maxTokens", self.max_tokens),
            "topKReturn": request_args.pop("topKReturn", self.top_k_return),
            "numResults": request_args.pop("numResults", self.num_results),
            "topP": request_args.pop("topP", self.top_p),
        }

        def _run_completion() -> Dict:
            post_str = self.host + "/" + self.engine + "/complete"
            res = requests.post(
                post_str,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_params,
            )
            return self.format_response(res.json())

        return _run_completion, request_params
