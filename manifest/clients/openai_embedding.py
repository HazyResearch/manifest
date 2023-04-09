"""OpenAI client."""
import copy
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import tiktoken

from manifest.clients.openai import OpenAIClient
from manifest.request import EmbeddingRequest

logger = logging.getLogger(__name__)

OPENAI_EMBEDDING_ENGINES = {
    "text-embedding-ada-002",
}


class OpenAIEmbeddingClient(OpenAIClient):
    """OpenAI client."""

    # User param -> (client param, default value)
    PARAMS = {
        "engine": ("model", "text-embedding-ada-002"),
    }
    REQUEST_CLS = EmbeddingRequest
    NAME = "openaiembedding"

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
        self.api_key = os.environ.get("OPENAI_API_KEY", connection_str)
        if self.api_key is None:
            raise ValueError(
                "OpenAI API key not set. Set OPENAI_API_KEY environment "
                "variable or pass through `client_connection`."
            )
        self.host = "https://api.openai.com/v1"
        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))
        if getattr(self, "engine") not in OPENAI_EMBEDDING_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. "
                f"Must be {OPENAI_EMBEDDING_ENGINES}."
            )

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return self.host + "/embeddings"

    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference."""
        return True

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
        if "data" not in response:
            raise ValueError(f"Invalid response: {response}")
        if "usage" in response:
            # Handle splitting the usages for batch requests
            if len(response["data"]) == 1:
                if isinstance(response["usage"], list):
                    response["usage"] = response["usage"][0]
                response["usage"] = [response["usage"]]
            else:
                # Try to split usage
                split_usage = self.split_usage(request, response["data"])
                if split_usage:
                    response["usage"] = split_usage
        return response

    def _format_request_for_embedding(self, request_params: Dict[str, Any]) -> Dict:
        """Format request params for embedding.

        Args:
            request_params: request params.

        Returns:
            formatted request params.
        """
        # Format for embedding model
        request_params = copy.deepcopy(request_params)
        prompt = request_params.pop("prompt")
        if isinstance(prompt, str):
            prompt_list = [prompt]
        else:
            prompt_list = prompt
        request_params["input"] = prompt_list
        return request_params

    def _format_request_from_embedding(self, response_dict: Dict[str, Any]) -> Dict:
        """Format response from embedding for standard response.

        Args:
            response_dict: response.

        Return:
            formatted response.
        """
        new_choices = []
        response_dict = copy.deepcopy(response_dict)
        for res in response_dict.pop("data"):
            new_choices.append({"array": np.array(res["embedding"])})
        response_dict["choices"] = new_choices
        return response_dict

    def _run_completion(
        self, request_params: Dict[str, Any], retry_timeout: int
    ) -> Dict:
        """Execute completion request.

        Args:
            request_params: request params.
            retry_timeout: retry timeout.

        Returns:
            response as dict.
        """
        # Format for embedding model
        request_params = self._format_request_for_embedding(request_params)
        response_dict = super()._run_completion(request_params, retry_timeout)
        # Reformat for text model
        response_dict = self._format_request_from_embedding(response_dict)
        return response_dict

    async def _arun_completion(
        self, request_params: Dict[str, Any], retry_timeout: int, batch_size: int
    ) -> Dict:
        """Async execute completion request.

        Args:
            request_params: request params.
            retry_timeout: retry timeout.
            batch_size: batch size for requests.

        Returns:
            response as dict.
        """
        # Format for embedding model
        request_params = self._format_request_for_embedding(request_params)
        response_dict = await super()._arun_completion(
            request_params, retry_timeout, batch_size
        )
        # Reformat for text model
        response_dict = self._format_request_from_embedding(response_dict)
        return response_dict

    def split_usage(self, request: Dict, choices: List[str]) -> List[Dict[str, int]]:
        """Split usage into list of usages for each prompt."""
        try:
            encoding = tiktoken.encoding_for_model(getattr(self, "engine"))
        except Exception:
            return []
        prompt = request["input"]
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        assert len(prompts) == len(choices)
        usages = []
        for pmt in prompts:
            pmt_tokens = len(encoding.encode(pmt))
            # No completion tokens for embedding models
            chc_tokens = 0
            usage = {
                "prompt_tokens": pmt_tokens,
                "completion_tokens": chc_tokens,
                "total_tokens": pmt_tokens + chc_tokens,
            }
            usages.append(usage)
        return usages
