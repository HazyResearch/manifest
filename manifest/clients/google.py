"""Google client."""
import logging
import os
import subprocess
from typing import Any, Dict, Optional, Type

from manifest.clients.client import Client
from manifest.request import LMRequest, Request

logger = logging.getLogger(__name__)

# https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/api-quickstart
GOOGLE_ENGINES = {
    "text-bison",
}


def get_project_id() -> Optional[str]:
    """Get project ID.

    Run
    `gcloud config get-value project`
    """
    try:
        project_id = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if project_id.stderr.decode("utf-8").strip():
            return None
        return project_id.stdout.decode("utf-8").strip()
    except Exception:
        return None


class GoogleClient(Client):
    """Google client."""

    # User param -> (client param, default value)
    PARAMS = {
        "engine": ("model", "text-bison"),
        "temperature": ("temperature", 1.0),
        "max_tokens": ("maxOutputTokens", 10),
        "top_p": ("topP", 1.0),
        "top_k": ("topK", 1),
        "batch_size": ("batch_size", 20),
    }
    REQUEST_CLS: Type[Request] = LMRequest
    NAME = "google"

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
        connection_parts = connection_str.split("::")
        if len(connection_parts) == 1:
            self.api_key = connection_parts[0]
            self.project_id = None
        elif len(connection_parts) == 2:
            self.api_key, self.project_id = connection_parts
        else:
            raise ValueError(
                "Invalid connection string. "
                "Must be either API_KEY or API_KEY::PROJECT_ID"
            )
        self.api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        if self.api_key is None:
            raise ValueError(
                "GoogleVertex API key not set. Set GOOGLE_API_KEY environment "
                "variable or pass through `client_connection`. This can be "
                "found by running `gcloud auth print-access-token`"
            )
        self.project_id = (
            self.project_id or os.environ.get("GOOGLE_PROJECT_ID") or get_project_id()
        )
        if self.project_id is None:
            raise ValueError("GoogleVertex project ID not set. Set GOOGLE_PROJECT_ID")
        self.host = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/us-central1/publishers/google/models"  # noqa: E501

        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))
        if getattr(self, "engine") not in GOOGLE_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {GOOGLE_ENGINES}."
            )

    def close(self) -> None:
        """Close the client."""
        pass

    def get_generation_url(self) -> str:
        """Get generation URL."""
        model = getattr(self, "engine")
        return self.host + f"/{model}:predict"

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return {"Authorization": f"Bearer {self.api_key}"}

    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference."""
        return True

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

    def preprocess_request_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess request params.

        Args:
            request: request params.

        Returns:
            request params.
        """
        # Refortmat the request params for google
        prompt = request.pop("prompt")
        if isinstance(prompt, str):
            prompt_list = [prompt]
        else:
            prompt_list = prompt
        google_request = {
            "instances": [{"prompt": prompt} for prompt in prompt_list],
            "parameters": request,
        }
        return super().preprocess_request_params(google_request)

    def postprocess_response(
        self, response: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate response as dict.

        Assumes response is dict
        {
            "predictions": [
                {
                    "safetyAttributes": {
                        "categories": ["Violent", "Sexual"],
                        "blocked": false,
                        "scores": [0.1, 0.1]
                    },
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
                    "text": prediction["content"],
                }
                for prediction in google_predictions
            ]
        }
        return super().postprocess_response(new_response, request)
