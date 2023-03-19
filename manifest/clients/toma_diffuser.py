"""TOMA client."""
import base64
import io
import logging
from typing import Any, Dict

import numpy as np
from PIL import Image

from manifest.clients.toma import TOMAClient
from manifest.request import DiffusionRequest

logger = logging.getLogger(__name__)

# Engines are dynamically instantiated from API
# but a few example engines are listed below.
TOMA_ENGINES = {
    "StableDiffusion",
}


class TOMADiffuserClient(TOMAClient):
    """TOMADiffuser client."""

    # User param -> (client param, default value)
    PARAMS = {
        "engine": ("model", "StableDiffusion"),
        "num_inference_steps": ("steps", 50),
        "height": ("height", 512),
        "width": ("width", 512),
        "n": ("n", 1),
        "guidance_scale": ("guidance_scale", 7.5),
    }
    REQUEST_CLS = DiffusionRequest  # type: ignore
    NAME = "tomadiffuser"

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        return {"model_name": "tomadiffuser", "engine": getattr(self, "engine")}

    def format_response(self, response: Dict, request: Dict) -> Dict[str, Any]:
        """
        Format response to dict.

        Args:
            response: response
            request: request

        Return:
            response as dict
        """
        return {
            "model": getattr(self, "engine"),
            "choices": [
                {
                    "array": np.array(
                        Image.open(
                            io.BytesIO(
                                base64.decodebytes(bytes(item["image_base64"], "utf-8"))
                            )
                        )
                    ),
                }
                for item in response["output"]["choices"]
            ],
        }
