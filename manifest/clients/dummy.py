"""Dummy client."""
import logging
from typing import Any, Dict, Optional

from manifest.clients.client import Client
from manifest.request import LMRequest, Request
from manifest.response import Response

logger = logging.getLogger(__name__)


class DummyClient(Client):
    """Dummy client."""

    # User param -> (client param, default value)
    PARAMS = {
        "n": ("num_results", 1),
    }
    REQUEST_CLS = LMRequest
    NAME = "dummy"

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to dummpy server.

        This is a dummy client that returns identity responses. Used for testing.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))

    def close(self) -> None:
        """Close the client."""
        pass

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return "dummy"

    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference."""
        return True

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return {}

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        return {"engine": "dummy"}

    def run_request(self, request: Request) -> Response:
        """
        Get request string function.

        Args:
            request: request.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        if isinstance(request.prompt, list):
            num_results = len(request.prompt)
        else:
            num_results = 1
        request_params = request.to_dict(self.PARAMS)

        response_dict = {
            "choices": [{"text": "hello"}]
            * int(request_params["num_results"])
            * num_results,
            "usage": [{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}]
            * int(request_params["num_results"])
            * num_results,
        }
        return Response(response_dict, False, request_params)

    async def arun_batch_request(self, request: Request) -> Response:
        """
        Get async request string function.

        Args:
            request: request.

        Returns:
            response.
        """
        return self.run_request(request)

    def get_score_prompt_request(
        self,
        request: Request,
    ) -> Response:
        """
        Get the logit score of the prompt via a forward pass of the model.

        Args:
            request: request.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        if isinstance(request.prompt, list):
            num_results = len(request.prompt)
        else:
            num_results = 1
        request_params = {"prompt": request.prompt}

        response_dict = {
            "choices": [
                {
                    "text": request.prompt
                    if isinstance(request.prompt, str)
                    else request.prompt[i],
                    "logprob": 0.3,
                }
                for i in range(num_results)
            ]
        }
        return Response(response_dict, False, request_params)
