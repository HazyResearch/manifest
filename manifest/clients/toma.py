"""TOMA client."""
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from manifest.clients.client import Client

logger = logging.getLogger(__name__)

TOMA_ENGINES = {
    "bloom",
    "glm-int8",
    "gpt-neox-20b",
    "opt-66b",
    "opt-175b",
    "glm",
    "stable_diffusion",
    "t0pp",
    "gpt-j-6b",
    "t5-11b",
    "glm-int4",
    "ul2",
}

# Engine -> request type
# Default is language-model-inference
TOMA_ENGINE_REQUEST_TYPE = {
    "stable_diffusion": "image-model-inference",
}

# User param -> (client param, default value)
TOMA_PARAMS = {
    "engine": ("model", "gpt-j-6b"),
    "temperature": ("temperature", 1.0),
    "max_tokens": ("max_tokens", 10),
    "n": ("n", 1),
    "p": ("top_p", 1.0),
    "stop_sequences": ("stop", []),
    "best_of": ("best_of", 1),
    "logprobs": ("logprobs", 0),
    "prompt_embedding": ("prompt_embedding", False),
    "echo": ("echo", False),
    "client_timeout": ("client_timeout", 120),  # seconds
}


class TOMAClient(Client):
    """TOMA client."""

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
        #         "variable or pass through `connection_str`."
        #     )

        for key in TOMA_PARAMS:
            setattr(self, key, client_args.pop(key, TOMA_PARAMS[key][1]))
        if getattr(self, "engine") not in TOMA_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {TOMA_ENGINES}."
            )
        model_heartbeats = self.get_model_heartbeats()
        model_heartbeat_threshold = 60
        logger.info(f"TOMA model heartbeats\n {json.dumps(model_heartbeats)}")
        if (
            model_heartbeats[getattr(self, "engine")]["last_ping"]
            > model_heartbeat_threshold
        ):
            raise ValueError(
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
        self.pending_jobs: List = []
        self.completed_jobs: List = []

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
        return {"model_name": "toma", "engine": getattr(self, "engine")}

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(TOMA_PARAMS.keys())

    def get_last_job_id(self) -> Optional[str]:
        """
        Get last job id.

        Returns:
            last job id.
        """
        if len(self.completed_jobs) > 0:
            return self.completed_jobs[-1]
        return None

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

    def format_response(self, response: Dict) -> Dict[str, Any]:
        """
        Format response to dict.

        Args:
            response: response

        Return:
            response as dict
        """
        return {
            "model": getattr(self, "engine"),
            "choices": [
                {
                    "text": item["text"],
                    # "logprobs": [],
                }
                for item in response["inference_result"][0]["choices"]
            ],
        }

    def get_response(self, job_id: str, retry_timeout: int) -> Dict[str, Any]:
        """
        Get response from job id.

        Will try up to `client_timeout` seconds to get response.

        Args:
            job_id: job id
            retry_timeout: retry timeout

        Returns:
            response as dict
        """
        final_res = None
        attempts = 0
        while True:
            ret = requests.get(f"{self.host}/job/{job_id}", json={"id": job_id}).json()
            attempts += 1
            if ret["status"] == "finished" or ret["status"] == "failed":
                final_res = ret["returned_payload"]
                break
            if attempts > retry_timeout:
                break
            time.sleep(1)
        if not final_res:
            raise RuntimeError(
                f"TOMA request timed out after {retry_timeout}s with {ret['status']}."
            )
        if "result" in final_res:
            return self.format_response(final_res["result"])
        else:
            raise RuntimeError(f"TOMA request failed with {final_res['message']}.")

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
        request_params = {
            "prompt": [query],
            "request_type": TOMA_ENGINE_REQUEST_TYPE.get(
                getattr(self, "engine"), "language-model-inference"
            ),
        }
        for key in TOMA_PARAMS:
            if key in ["client_timeout"]:
                # These are not passed to the TOMA API
                continue
            request_params[TOMA_PARAMS[key][0]] = request_args.pop(
                key, getattr(self, key)
            )

        retry_timeout = request_args.pop(
            "client_timeout", getattr(self, "client_timeout")
        )

        # num_returns is for image-model-inference
        if request_params["request_type"] == "image-model-inference":
            request_params["num_returns"] = request_params["n"]

        def _run_completion() -> Dict:
            post_str = self.host + "/jobs"
            res = requests.post(
                post_str,
                # headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "type": "general",
                    "payload": request_params,
                    "returned_payload": {},
                    "status": "submitted",
                    "source": "dalle",
                },
            ).json()
            job_id = res["id"]
            # TODO: ideally just submit the jobs and then fetch results in parallel
            self.pending_jobs.append(job_id)
            job_id = self.pending_jobs.pop()
            final_res = self.get_response(job_id, retry_timeout)
            self.completed_jobs.append(job_id)
            return final_res

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
        raise NotImplementedError("TOMA does not support choice logit request.")
