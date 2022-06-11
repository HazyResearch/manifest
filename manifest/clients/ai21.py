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
            "engine": kwargs.get("engine", self.engine),
            "prompt": query,
            "temperature": kwargs.get("temperature", self.temperature),
            "maxTokens": kwargs.get("maxTokens", self.max_tokens),
            "topKReturn": kwargs.get("topKReturn", self.top_k_return),
            "numResults": kwargs.get("numResults", self.num_results),
            "topP": kwargs.get("topP", self.top_p),
        }

        def _run_completion() -> Dict:
            post_str = self.host + "/" + self.engine + "/complete"
            print(self.api_key)
            print(post_str)
            print("https://api.ai21.com/studio/v1/j1-large/complete")
            print(request_params)
            res = requests.post(
                post_str,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_params,
            )
            return res.json()

        return _run_completion, request_params
