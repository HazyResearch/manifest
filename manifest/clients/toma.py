"""TOMA client."""
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import requests

from manifest.clients.client import Client
from manifest.request import LMRequest

logger = logging.getLogger(__name__)

# Engines are dynamically instantiated from API
# but a few example engines are listed below.
TOMA_ENGINES = {
    "Together-gpt-JT-6B-v1",
}


class TOMAClient(Client):
    """TOMA client."""

    # User param -> (client param, default value)
    PARAMS = {
        "engine": ("model", "Together-gpt-JT-6B-v1"),
        "temperature": ("temperature", 0.1),
        "max_tokens": ("max_tokens", 32),
        # n is depricated with new API but will come back online soon
        # "n": ("n", 1),
        "top_p": ("top_p", 0.9),
        "top_k": ("top_k", 40),
        "stop_sequences": ("stop", []),
    }
    REQUEST_CLS = LMRequest
    NAME = "toma"

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the TOMA url.

        Arsg:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.host = os.environ.get("TOMA_URL", None)
        if not self.host:
            raise ValueError("TOMA_URL environment variable not set.")
        # self.api_key = os.environ.get("TOMA_API_KEY", connection_str)
        # if self.api_key is None:
        #     raise ValueError(
        #         "TOMA API key not set. Set TOMA_API_KEY environment "
        #         "variable or pass through `client_connection`."
        #     )

        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))

        # Not functioning yet in new TOMA API. Will come back online soon.
        """
        model_heartbeats = self.get_model_heartbeats()
        if getattr(self, "engine") not in model_heartbeats.keys():
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. "
                f"Must be {model_heartbeats.keys()}."
            )
        model_heartbeat_threshold = 120
        logger.info(f"TOMA model heartbeats\n {json.dumps(model_heartbeats)}")
        if (
            model_heartbeats[getattr(self, "engine")]["last_ping"]
            > model_heartbeat_threshold
        ):
            logger.warning(
                f"Model {getattr(self, 'engine')} has not been pinged in "
                f"{model_heartbeats[getattr(self, 'engine')]} seconds."
            )
        if model_heartbeats[getattr(self, "engine")]["expected_runtime"] > getattr(
            self, "client_timeout"
        ):
            logger.warning(
                f"Model {getattr(self, 'engine')} has expected runtime "
                f"{model_heartbeats[getattr(self, 'engine')]['expected_runtime']} "
                f"and may take longer than {getattr(self, 'client_timeout')} "
                "seconds to respond. Increase client_timeout "
                "to avoid timeout."
            )
        """

    def close(self) -> None:
        """Close the client."""
        pass

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return self.host + "/inference"

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return {}

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
        return {"model_name": "toma", "engine": getattr(self, "engine")}

    def get_model_heartbeats(self) -> Dict[str, Dict]:
        """
        Get TOMA models and their last ping time.

        Some TOMA models are not loaded and will not response.

        Returns:
            model name to time since last ping (sec).
        """
        res = requests.get(self.host + "/model_statuses").json()
        heartbeats = {}
        for mod in res:
            mod_time = datetime.fromisoformat(mod["last_heartbeat"])
            now = datetime.now(mod_time.tzinfo)
            heartbeats[mod["name"]] = {
                "last_ping": (now - mod_time).total_seconds(),
                "expected_runtime": mod["expected_runtime"],
            }
        return heartbeats

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
                    "text": item["text"],
                    # "token_logprobs": [],
                }
                for item in response["output"]["choices"]
            ],
        }
