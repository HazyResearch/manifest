"""OpenAI client."""
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import openai

from manifest.clients import Response
from manifest.clients.client import Client

logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

OPENAI_ENGINES = {
    "text-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
}


class OpenAIClient(Client):
    """OpenAI client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        engine: Optional[str] = "text-ada-001",
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = 10,
        top_p: Optional[int] = 1,
        frequency_penalty: Optional[int] = 0,
        presence_penalty: Optional[int] = 0,
        n: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:
        """
        Connect to the OpenAI server.

        connection_str is passed as default OPENAI_API_KEY if variable not set.
        """
        openai.api_key = os.environ.get("OPENAI_API_KEY", connection_str)
        if openai.api_key is None:
            raise ValueError(
                "OpenAI API key not set. Set OPENAI_API_KEY environment ",
                "svariable or pass through `connection_str`.",
            )
        self.engine = engine
        if self.engine not in OPENAI_ENGINES:
            raise ValueError(f"Invalid engine {self.engine}. Must be {OPENAI_ENGINES}.")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.n = n

    def close(self) -> None:
        """Close the client."""
        pass

    def get_request(
        self, query: str, **kwargs: Any
    ) -> Tuple[Callable[[], Response], Dict]:
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
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", self.frequency_penalty
            ),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
            "n": kwargs.get("n", self.n),
        }

        def _run_completion() -> Response:
            try:
                return Response(openai.Completion.create(**request_params))
            except openai.error.OpenAIError as e:
                logger.error(e)
                raise e

        return _run_completion, request_params
