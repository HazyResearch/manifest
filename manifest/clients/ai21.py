"""AI21 client."""
import logging
import os
from typing import Any, Dict, Optional

from manifest.clients.client import Client
from manifest.request import LMRequest

logger = logging.getLogger(__name__)

AI21_ENGINES = {
    "j1-jumbo",
    "j1-grande",
    "j1-large",
}


class AI21Client(Client):
    """AI21Client client."""

    # User param -> (client param, default value)
    PARAMS = {
        "engine": ("engine", "j1-large"),
        "temperature": ("temperature", 1.0),
        "max_tokens": ("maxTokens", 10),
        "top_k": ("topKReturn", 0),
        "n": ("numResults", 1),
        "top_p": ("topP", 1.0),
        "stop_sequences": ("stopSequences", []),
    }
    REQUEST_CLS = LMRequest
    NAME = "ai21"

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
        self.api_key = connection_str or os.environ.get("AI21_API_KEY")
        if self.api_key is None:
            raise ValueError(
                "AI21 API key not set. Set AI21_API_KEY environment "
                "variable or pass through `client_connection`."
            )

        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))
        if getattr(self, "engine") not in AI21_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {AI21_ENGINES}."
            )

    def close(self) -> None:
        """Close the client."""
        pass

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return self.host + "/" + getattr(self, "engine") + "/complete"

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return {"Authorization": f"Bearer {self.api_key}"}

    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference."""
        return False

    def supports_streaming_inference(self) -> bool:
        """Return whether the client supports streaming inference.

        Override in child client class.
        """
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

    def postprocess_response(self, response: Dict, request: Dict) -> Dict[str, Any]:
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
                    "text": item["data"]["text"],
                    "token_logprobs": item["data"]["tokens"],
                }
                for item in response["completions"]
            ],
        }
