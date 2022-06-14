"""OpenAI client."""
import logging
import os
import sys
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

from manifest.clients.client import Client

crfm_code_dir = os.environ.get("CRFM_CODE_DIR", "/home/code/benchmarking")
sys.path.append(crfm_code_dir)

from src.common.authentication import Authentication  # type: ignore
from src.common.request import Request, RequestResult  # type: ignore
from src.proxy.remote_service import RemoteService  # type: ignore

logger = logging.getLogger(__name__)

CRFM_ENGINES = {
    "ai21/j1-jumbo",
    "ai21/j1-grande",
    "ai21/j1-large",
}


class CRFMClient(Client):
    """CRFMClient client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the CRFM endpoint.

        connection_str is passed as default CRFM_API_KEY if variable not set.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.service = RemoteService("https://crfm-models.stanford.edu")
        api_key = os.environ.get("CRFM_API_KEY", connection_str)
        if api_key is None:
            raise ValueError(
                "CRFM API key not set. Set CRFM_API_KEY environment "
                "variable or pass through `connection_str`."
            )
        self.auth = Authentication(api_key=api_key)
        self.engine = client_args.pop("engine", "ai21/j1-large")
        if self.engine not in CRFM_ENGINES:
            raise ValueError(f"Invalid engine {self.engine}. Must be {CRFM_ENGINES}.")
        self.temperature = client_args.pop("temperature", 0.0)
        self.max_tokens = client_args.pop("max_tokens", 10)
        self.top_k_per_token = client_args.pop("top_k_per_token", 1)
        self.num_completions = client_args.pop("num_completions", 1)
        self.stop_sequences = client_args.pop("stop_sequences", [])
        self.top_p = client_args.pop("top_p", 1.0)
        self.presence_penalty = client_args.pop("presence_penalty", 1.0)
        self.frequency_penalty = client_args.pop("frequency_penalty", 1.0)

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
        return {"model_name": "crfm", "engine": self.engine}

    def format_response(self, response: RequestResult) -> Dict[str, Any]:
        """
        Format RequestResult to dict.

        Args:
            response: RequestResult

        Return:
            response as dict
        """
        return {
            "id": str(uuid.uuid4()),
            "object": "text_completion",
            "model": self.engine,
            "choices": [
                {
                    "text": text.text,
                    # TODO: Add in more metadata for HF models
                    # "logprobs": {
                    #     "tokens": result["tokens"],
                    #     "token_logprobs": result["token_scores"],
                    #     "text_offset": result["text_offset"],
                    #     "top_logprobs": result["top_logprobs"],
                    #     "finish_reason": "length",
                    # },
                }
                for text in response.completions
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
            "model": request_args.pop("engine", self.engine),
            "prompt": query,
            "temperature": request_args.pop("temperature", self.temperature),
            "max_tokens": request_args.pop("max_tokens", self.max_tokens),
            "top_k_per_token": request_args.pop(
                "top_k_per_token", self.top_k_per_token
            ),
            "num_completions": request_args.pop(
                "num_completions", self.num_completions
            ),
            "stop_sequences": request_args.pop("stop_sequences", self.stop_sequences),
            "top_p": request_args.pop("top_p", self.top_p),
            "presence_penalty": request_args.pop(
                "presence_penalty", self.presence_penalty
            ),
            "frequency_penalty": request_args.pop(
                "frequency_penalty", self.frequency_penalty
            ),
        }

        def _run_completion() -> Dict:
            request = Request(**request_params)
            request_result = self.service.make_request(self.auth, request)
            return self.format_response(request_result)

        return _run_completion, request_params
