"""OpenAIChat client."""
import copy
import logging
import os
from typing import Any, Dict, Optional

from manifest.clients.openai import OpenAIClient
from manifest.request import LMRequest

logger = logging.getLogger(__name__)

# List from https://platform.openai.com/docs/models/model-endpoint-compatibility
OPENAICHAT_ENGINES = {"gpt-3.5-turbo", "gpt-4", "gpt-4-32k"}


class OpenAIChatClient(OpenAIClient):
    """OpenAI Chat client."""

    # User param -> (client param, default value)
    PARAMS = {
        "engine": ("model", "gpt-3.5-turbo"),
        "temperature": ("temperature", 1.0),
        "max_tokens": ("max_tokens", 10),
        "n": ("n", 1),
        "top_p": ("top_p", 1.0),
        "stop_sequences": ("stop", None),  # OpenAI doesn't like empty lists
        "presence_penalty": ("presence_penalty", 0.0),
        "frequency_penalty": ("frequency_penalty", 0.0),
    }
    REQUEST_CLS = LMRequest
    NAME = "openaichat"
    IS_CHAT = True

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
        if getattr(self, "engine") not in OPENAICHAT_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. "
                f"Must be {OPENAICHAT_ENGINES}."
            )

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return self.host + "/chat/completions"

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

    def _format_request_for_chat(self, request_params: Dict[str, Any]) -> Dict:
        """Format request params for chat.

        Args:
            request_params: request params.

        Returns:
            formatted request params.
        """
        # Format for chat model
        request_params = copy.deepcopy(request_params)
        prompt = request_params.pop("prompt")
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and isinstance(prompt[0], str):
            prompt_list = prompt
            messages = [{"role": "user", "content": prompt} for prompt in prompt_list]
        elif isinstance(prompt, list) and isinstance(prompt[0], dict):
            for pmt_dict in prompt:
                if "role" not in pmt_dict or "content" not in pmt_dict:
                    raise ValueError(
                        "Prompt must be list of dicts with 'role' and 'content' "
                        f"keys. Got {prompt}."
                    )
            messages = prompt
        else:
            raise ValueError(
                "Prompt must be string, list of strings, or list of dicts."
                f"Got {prompt}"
            )
        request_params["messages"] = messages
        return request_params

    def _format_request_from_chat(self, response_dict: Dict[str, Any]) -> Dict:
        """Format response for standard response from chat.

        Args:
            response_dict: response.

        Return:
            formatted response.
        """
        new_choices = []
        response_dict = copy.deepcopy(response_dict)
        for message in response_dict["choices"]:
            new_choices.append({"text": message["message"]["content"]})
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
        # Format for chat model
        request_params = self._format_request_for_chat(request_params)
        response_dict = super()._run_completion(request_params, retry_timeout)
        # Reformat for text model
        response_dict = self._format_request_from_chat(response_dict)
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
        # Format for chat model
        request_params = self._format_request_for_chat(request_params)
        response_dict = await super()._arun_completion(
            request_params, retry_timeout, batch_size
        )
        # Reformat for text model
        response_dict = self._format_request_from_chat(response_dict)
        return response_dict
