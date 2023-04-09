"""Cohere client."""

import logging
import os
from typing import Any, Dict, Optional

from manifest.clients.client import Client
from manifest.request import LMRequest

logger = logging.getLogger(__name__)

COHERE_MODELS = {"small", "medium", "large", "xlarge"}


class CohereClient(Client):
    """Cohere client."""

    # Params are defined in https://docs.cohere.ai/generate-reference
    PARAMS = {
        "engine": ("model", "xlarge"),
        "max_tokens": ("max_tokens", 20),
        "temperature": ("temperature", 0.75),
        "n": ("num_generations", 1),
        "top_k": ("k", 0),
        "top_p": ("p", 0.75),
        "frequency_penalty": ("frequency_penalty", 0.0),
        "presence_penalty": ("presence_penalty", 0.0),
        "stop_sequences": ("stop_sequences", None),
    }
    REQUEST_CLS = LMRequest
    NAME = "cohere"

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
                "variable or pass through `client_connection`."
            )
        self.host = "https://api.cohere.ai"
        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))
        if getattr(self, "engine") not in COHERE_MODELS:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {COHERE_MODELS}."
            )

    def close(self) -> None:
        """Close the client."""

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return self.host + "/generate"

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return {
            "Cohere-Version": "2021-11-08",
            "Authorization": f"Bearer {self.api_key}",
        }

    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference."""
        return False

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        return {"model_name": self.NAME, "engine": getattr(self, "engine")}

    def format_response(self, response: Dict, request: Dict) -> Dict[str, Any]:
        """
        Format response to dict.

        Args:
            response: response
            request: request

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
                    "token_logprobs": item.get("token_likelihoods", None),
                }
                for item in response["generations"]
            ],
        }
