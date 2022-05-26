"""OpenAI response."""

import time
import uuid
from typing import Any, Dict


class OpenAIResponse:
    """OpenAI response."""

    def __init__(self, results: list) -> None:
        """Initialize response."""
        self.results = results
        self.response_id = str(uuid.uuid4())
        self.created = int(time.time())

    def __dict__(self) -> Dict[str, Any]:  # type: ignore
        """Return dictionary representation of response."""
        return {
            "id": self.response_id,
            "object": "text_completion",
            "created": self.created,
            "model": "flask_model",
            "choices": [
                {
                    "text": result,
                    # TODO: Add in more metadata for HF models
                    # "logprobs": {
                    #     "tokens": result["tokens"],
                    #     "token_logprobs": result["token_scores"],
                    #     "text_offset": result["text_offset"],
                    #     "top_logprobs": result["top_logprobs"],
                    #     "finish_reason": "length",
                    # },
                }
                for result in self.results
            ],
        }
