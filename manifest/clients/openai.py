"""OpenAI client."""
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import openai

from manifest.clients.client import Client

logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

OPENAI_ENGINES = {
    "text-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
    "code-davinci-002",
    "code-cushman-001",
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
        openai.api_key = os.environ.get("OPENAI_API_KEY", connection_str)
        if openai.api_key is None:
            raise ValueError(
                "OpenAI API key not set. Set OPENAI_API_KEY environment "
                "variable or pass through `connection_str`."
            )
        self.engine = client_args.pop("engine", "text-davinci-002")
        if self.engine not in OPENAI_ENGINES:
            raise ValueError(f"Invalid engine {self.engine}. Must be {OPENAI_ENGINES}.")
        self.temperature = client_args.pop("temperature", 0.0)
        self.max_tokens = client_args.pop("max_tokens", 10)
        self.top_p = client_args.pop("top_p", 1.0)
        self.logprobs = client_args.pop("logprobs", None)
        self.best_of = client_args.pop("best_of", 1)
        self.frequency_penalty = client_args.pop("frequency_penalty", 0.0)
        self.presence_penalty = client_args.pop("presence_penalty", 0.0)
        self.n = client_args.pop("n", 1)

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
        return {"model_name": "openai", "engine": self.engine}

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
            "max_tokens": request_args.pop("max_tokens", self.max_tokens),
            "top_p": request_args.pop("top_p", self.top_p),
            "frequency_penalty": request_args.pop(
                "frequency_penalty", self.frequency_penalty
            ),
            "logprobs": request_args.pop("logprobs", self.logprobs),
            "best_of": request_args.pop("best_of", self.best_of),
            "presence_penalty": request_args.pop(
                "presence_penalty", self.presence_penalty
            ),
            "n": request_args.pop("n", self.n),
        }

        def _run_completion() -> Dict:
            try:
                return openai.Completion.create(**request_params)
            except openai.error.OpenAIError as e:
                logger.error(e)
                raise e

        return _run_completion, request_params
