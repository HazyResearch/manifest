"""OpenAI client."""
import copy
import logging
import os
from typing import Any, Dict, Optional, Type

from manifest.clients.google import GoogleClient, get_project_id
from manifest.request import LMRequest, Request

logger = logging.getLogger(__name__)

# https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/api-quickstart
GOOGLE_ENGINES = {
    "chat-bison",
}


class GoogleChatClient(GoogleClient):
    """GoogleChat client."""

    # User param -> (client param, default value)
    PARAMS = {
        "engine": ("model", "chat-bison"),
        "temperature": ("temperature", 1.0),
        "max_tokens": ("maxOutputTokens", 10),
        "top_p": ("topP", 1.0),
        "top_k": ("topK", 1),
        "batch_size": ("batch_size", 20),
    }
    REQUEST_CLS: Type[Request] = LMRequest
    NAME = "googlechat"
    IS_CHAT = True

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the GoogleVertex API.

        connection_str is passed as default GOOGLE_API_KEY if variable not set.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.api_key = os.environ.get("GOOGLE_API_KEY", connection_str)
        if self.api_key is None:
            raise ValueError(
                "GoogleVertex API key not set. Set GOOGLE_API_KEY environment "
                "variable or pass through `client_connection`. This can be "
                "found by running `gcloud auth print-access-token`"
            )
        self.project_id = os.environ.get("GOOGLE_PROJECT_ID") or get_project_id()
        if self.project_id is None:
            raise ValueError("GoogleVertex project ID not set. Set GOOGLE_PROJECT_ID")
        self.host = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/us-central1/publishers/google/models"  # noqa: E501

        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))
        if getattr(self, "engine") not in GOOGLE_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {GOOGLE_ENGINES}."
            )

    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference."""
        return False

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
            messages = [{"author": "user", "content": prompt}]
        elif isinstance(prompt, list) and isinstance(prompt[0], str):
            prompt_list = prompt
            messages = [{"author": "user", "content": prompt} for prompt in prompt_list]
        elif isinstance(prompt, list) and isinstance(prompt[0], dict):
            for pmt_dict in prompt:
                if "author" not in pmt_dict or "content" not in pmt_dict:
                    raise ValueError(
                        "Prompt must be list of dicts with 'author' and 'content' "
                        f"keys. Got {prompt}."
                    )
            messages = prompt
        else:
            raise ValueError(
                "Prompt must be string, list of strings, or list of dicts."
                f"Got {prompt}"
            )
        new_request = {
            "instances": [{"messages": messages}],
            "parameters": request_params,
        }
        return new_request

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
        response_dict = super(GoogleClient, self)._run_completion(
            request_params, retry_timeout
        )
        # Validate response handles the reformatting
        return response_dict

    async def _arun_completion(
        self, request_params: Dict[str, Any], retry_timeout: int
    ) -> Dict:
        """Async execute completion request.

        Args:
            request_params: request params.
            retry_timeout: retry timeout.

        Returns:
            response as dict.
        """
        # Format for chat model
        request_params = self._format_request_for_chat(request_params)
        response_dict = await super(GoogleClient, self)._arun_completion(
            request_params, retry_timeout
        )
        # Validate response handles the reformatting
        return response_dict

    def validate_response(self, response: Dict, request: Dict) -> Dict[str, Any]:
        """
        Validate response as dict.

        Assumes response is dict
        {
            "candidates": [
                {
                    "safetyAttributes": {
                        "categories": ["Violent", "Sexual"],
                        "blocked": false,
                        "scores": [0.1, 0.1]
                    },
                    "author": "1",
                    "content": "SELECT * FROM "WWW";"
                }
            ]
        }

        Args:
            response: response
            request: request

        Return:
            response as dict
        """
        google_predictions = response.pop("predictions")
        new_response = {
            "choices": [
                {
                    "text": prediction["candidates"][0]["content"],
                }
                for prediction in google_predictions
            ]
        }
        return super(GoogleClient, self).validate_response(new_response, request)
