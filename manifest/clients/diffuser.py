"""Diffuser client."""
import logging
from typing import Any, Dict, Optional

import numpy as np
import requests

from manifest.clients.client import Client
from manifest.request import DiffusionRequest

logger = logging.getLogger(__name__)


class DiffuserClient(Client):
    """Diffuser client."""

    # User param -> (client param, default value)
    PARAMS = {
        "num_inference_steps": ("num_inference_steps", 50),
        "height": ("height", 512),
        "width": ("width", 512),
        "n": ("num_images_per_prompt", 1),
        "guidance_scale": ("guidance_scale", 7.5),
        "eta": ("eta", 0.0),
    }
    REQUEST_CLS = DiffusionRequest
    NAME = "diffuser"

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the Diffuser url.

        Arsg:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.host = connection_str.rstrip("/")
        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))
        self.model_params = self.get_model_params()

    def to_numpy(self, image: np.ndarray) -> np.ndarray:
        """Convert a numpy image to a PIL image.

        Adapted from https://github.com/huggingface/diffusers/blob/src/diffusers/pipelines/pipeline_utils.py#L808   # noqa: E501
        """
        image = (image * 255).round().astype("uint8")
        return image

    def close(self) -> None:
        """Close the client."""
        pass

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return self.host + "/completions"

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return {}

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
        res = requests.post(self.host + "/params").json()
        res["client_name"] = self.NAME
        return res

    def postprocess_response(self, response: Dict, request: Dict) -> Dict[str, Any]:
        """
        Format response to dict.

        Args:
            response: response
            request: request

        Return:
            response as dict
        """
        # Convert array to np.array
        for choice in response["choices"]:
            choice["array"] = self.to_numpy(np.array(choice["array"]))
        return response
