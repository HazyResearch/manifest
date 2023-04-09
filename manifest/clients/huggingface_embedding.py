"""Hugging Face client."""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests

from manifest.clients.client import Client
from manifest.request import EmbeddingRequest

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingClient(Client):
    """HuggingFaceEmbedding client."""

    # User param -> (client param, default value)
    PARAMS: Dict[str, Tuple[str, Any]] = {}
    REQUEST_CLS = EmbeddingRequest
    NAME = "huggingfaceembedding"

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the HuggingFace url.

        Arsg:
            connection_str: connection string.
            client_args: client arguments.
        """
        if not connection_str:
            raise ValueError("Must provide connection string")
        self.host = connection_str.rstrip("/")
        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))

    def close(self) -> None:
        """Close the client."""
        pass

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return self.host + "/embed"

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

    def format_response(self, response: Dict, request: Dict) -> Dict[str, Any]:
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
            choice["array"] = np.array(choice["array"])
        return response
