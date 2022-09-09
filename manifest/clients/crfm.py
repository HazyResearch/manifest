"""OpenAI client."""
import logging
import os
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# User param -> (client param, default value)
CRFM_PARAMS = {
    "engine": ("engine", "ai21/j1-jumbo"),
    "temperature": ("temperature", 1.0),
    "max_tokens": ("max_tokens", 10),
    "n": ("num_completions", 1),
    "top_p": ("top_p", 1.0),
    "top_k_return": ("top_k_per_token", 1),
    "stop_sequences": ("stop_sequences", []),
    "presence_penalty": ("presence_penalty", 0.0),
    "frequency_penalty": ("frequency_penalty", 0.0),
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
        for key in CRFM_PARAMS:
            setattr(self, key, client_args.pop(key, CRFM_PARAMS[key][1]))
        if getattr(self, "engine") not in CRFM_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {CRFM_ENGINES}."
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
        return {"model_name": "crfm", "engine": getattr(self, "engine")}

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(CRFM_PARAMS.keys())

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
            "model": getattr(self, "engine"),
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
        request_params = {"prompt": query}
        for key in CRFM_PARAMS:
            request_params[CRFM_PARAMS[key][0]] = request_args.pop(
                key, getattr(self, key)
            )
        del request_params["engine"]

        def _run_completion() -> Dict:
            request = Request(**request_params)
            request_result = self.service.make_request(self.auth, request)
            return self.format_response(request_result)

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
        raise NotImplementedError("CRFM does not support choice logit request.")
