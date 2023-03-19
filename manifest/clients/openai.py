"""OpenAI client."""
import logging
import os
from typing import Any, Dict, List, Optional

import tiktoken

from manifest.clients.client import Client
from manifest.request import LMRequest

logger = logging.getLogger(__name__)

OPENAI_ENGINES = {
    "text-davinci-003",
    "text-davinci-002",
    "text-davinci-001",
    "davinci",
    "curie",
    "ada",
    "babbage",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
    "code-davinci-002",
    "code-cushman-001",
}


class OpenAIClient(Client):
    """OpenAI client."""

    # User param -> (client param, default value)
    PARAMS = {
        "engine": ("model", "text-davinci-003"),
        "temperature": ("temperature", 1.0),
        "max_tokens": ("max_tokens", 10),
        "n": ("n", 1),
        "top_p": ("top_p", 1.0),
        "top_k": ("best_of", 1),
        "stop_sequences": ("stop", None),  # OpenAI doesn't like empty lists
        "presence_penalty": ("presence_penalty", 0.0),
        "frequency_penalty": ("frequency_penalty", 0.0),
    }
    REQUEST_CLS = LMRequest
    NAME = "openai"

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
        if getattr(self, "engine") not in OPENAI_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {OPENAI_ENGINES}."
            )

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
        return {"Authorization": f"Bearer {self.api_key}"}

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
        return {"model_name": "openai", "engine": getattr(self, "engine")}

    def split_usage(self, request: Dict, choices: List[str]) -> List[Dict[str, int]]:
        """Split usage into list of usages for each prompt."""
        try:
            encoding = tiktoken.encoding_for_model(getattr(self, "engine"))
        except Exception:
            return []
        prompt = request["prompt"]
        # If n > 1 and prompt is a string, we need to split it into a list
        if isinstance(prompt, str):
            prompts = [prompt] * len(choices)
        else:
            prompts = prompt
        assert len(prompts) == len(choices)
        usages = []
        for pmt, chc in zip(prompts, choices):
            pmt_tokens = len(encoding.encode(pmt))
            chc_tokens = len(encoding.encode(chc["text"]))  # type: ignore
            usage = {
                "prompt_tokens": pmt_tokens,
                "completion_tokens": chc_tokens,
                "total_tokens": pmt_tokens + chc_tokens,
            }
            usages.append(usage)
        return usages
