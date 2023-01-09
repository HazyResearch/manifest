"""HELM client."""
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

from helm.common.authentication import Authentication
from helm.common.request import Request as HELMRequest
from helm.proxy.services.remote_service import RemoteService

from manifest.clients.client import Client
from manifest.request import LMRequest, Request

logger = logging.getLogger(__name__)

HELM_ENGINES = {
    "ai21/j1-jumbo" "ai21/j1-grande",
    "ai21/j1-grande-v2-beta",
    "ai21/j1-large",
    "AlephAlpha/luminous-base",
    "AlephAlpha/luminous-extended",
    "AlephAlpha/luminous-supreme",
    "anthropic/stanford-online-all-v4-s3",
    "together/bloom",
    "together/t0pp",
    "cohere/xlarge-20220609",
    "cohere/xlarge-20221108",
    "cohere/large-20220720",
    "cohere/medium-20220720",
    "cohere/medium-20221108",
    "cohere/small-20220720",
    "together/gpt-j-6b",
    "together/gpt-neox-20b",
    "gooseai/gpt-neo-20b",
    "gooseai/gpt-j-6b",
    "huggingface/gpt-j-6b",
    "together/t5-11b",
    "together/ul2",
    "huggingface/gpt2",
    "openai/davinci",
    "openai/curie",
    "openai/babbage",
    "openai/ada",
    "openai/text-davinci-003",
    "openai/text-davinci-002",
    "openai/text-davinci-001",
    "openai/text-curie-001",
    "openai/text-babbage-001",
    "openai/text-ada-001",
    "openai/code-davinci-002",
    "openai/code-davinci-001",
    "openai/code-cushman-001",
    "openai/chat-gpt",
    "openai/text-similarity-davinci-001",
    "openai/text-similarity-curie-001",
    "openai/text-similarity-babbage-001",
    "openai/text-similarity-ada-001",
    "together/opt-175b",
    "together/opt-66b",
    "microsoft/TNLGv2_530B",
    "microsoft/TNLGv2_7B",
    "together/Together-gpt-JT-6B-v1",
    "together/glm",
    "together/yalm",
}


class HELMClient(Client):
    """HELM client."""

    # User param -> (client param, default value)
    PARAMS = {
        "engine": ("model", "openai/text-davinci-002"),
        "temperature": ("temperature", 1.0),
        "max_tokens": ("max_tokens", 10),
        "n": ("num_completions", 1),
        "top_p": ("top_p", 1.0),
        "top_k": ("top_k_per_token", 1),
        "stop_sequences": ("stop_sequences", None),  # HELM doesn't like empty lists
        "presence_penalty": ("presence_penalty", 0.0),
        "frequency_penalty": ("frequency_penalty", 0.0),
        "client_timeout": ("client_timeout", 60),  # seconds
    }
    REQUEST_CLS = LMRequest

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Create a HELM instance.

        connection_str is passed as default HELM_API_KEY if variable not set.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.api_key = os.environ.get("HELM_API_KEY", connection_str)
        if self.api_key is None:
            raise ValueError(
                "HELM API key not set. Set HELM_API_KEY environment "
                "variable or pass through `client_connection`."
            )
        self._helm_auth = Authentication(api_key=self.api_key)
        self._help_api = RemoteService("https://crfm-models.stanford.edu")
        self._helm_account = self._help_api.get_account(self._helm_auth)
        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))
        if getattr(self, "engine") not in HELM_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {HELM_ENGINES}."
            )

    def close(self) -> None:
        """Close the client."""
        self._help_api = None

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return ""

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return ""

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
        return {"model_name": "HELM", "engine": getattr(self, "engine")}

    def get_request(self, request: Request) -> Tuple[Callable[[], Dict], Dict]:
        """
        Get request string function.

        Args:
            request: request.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        if isinstance(request.prompt, list):
            raise ValueError("HELM does not support batch inference.")

        request_params = request.to_dict(self.PARAMS)

        def _run_completion() -> Dict:
            try:
                request = HELMRequest(**request_params)
                request_result = self._help_api.make_request(self._helm_auth, request)
            except Exception as e:
                logger.error(f"HELM error {e}.")
                raise e
            return self.format_response(request_result.__dict__())

        return _run_completion, request_params
