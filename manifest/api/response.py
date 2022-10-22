"""Response."""

import time
import uuid
from typing import Any, Dict, List


class Response:
    """Response."""

    def __init__(self, results: List[Dict[str, Any]], response_type: str) -> None:
        """Initialize response."""
        self.results = results
        self.response_type = response_type
        if self.response_type not in {"text_completion", "choice_selection"}:
            raise ValueError(
                f"Invalid response type: {self.response_type}. "
                "Must be one of: text_completion, choice_selection."
            )
        self.response_id = str(uuid.uuid4())
        self.created = int(time.time())

    def __dict__(self) -> Dict[str, Any]:  # type: ignore
        """Return dictionary representation of response."""
        return {
            "id": self.response_id,
            "object": self.response_type,
            "created": self.created,
            "model": "flask_model",
            "choices": [
                {
                    "text": result["text"],
                    "text_logprob": result["text_logprob"],
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
