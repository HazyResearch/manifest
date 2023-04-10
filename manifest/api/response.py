"""Response."""

import time
import uuid
from typing import Any, Dict, List


class ModelResponse:
    """ModelResponse."""

    def __init__(self, results: List[Dict[str, Any]], response_type: str) -> None:
        """Initialize response."""
        self.results = results
        self.response_type = response_type
        if self.response_type not in {
            "text_completion",
            "prompt_logit_score",
            "image_generation",
            "embedding_generation",
        }:
            raise ValueError(
                f"Invalid response type: {self.response_type}. "
                "Must be one of: text_completion, prompt_logit_score, "
                "image_generation, embedding_generation."
            )
        self.response_id = str(uuid.uuid4())
        self.created = int(time.time())

    def __dict__(self) -> Dict[str, Any]:  # type: ignore
        """Return dictionary representation of response."""
        key = (
            "text"
            if self.response_type
            not in {"prompt_logit_score", "image_generation", "embedding_generation"}
            else "array"
        )
        return {
            "id": self.response_id,
            "object": self.response_type,
            "created": self.created,
            "model": "flask_model",
            "choices": [
                {
                    key: result[key],
                    "logprob": result["logprob"],
                    "tokens": result["tokens"],
                    "token_logprobs": result["token_logprobs"],
                }
                if key == "text"
                else {
                    key: result[key].tolist(),
                    "logprob": result["logprob"],
                }
                for result in self.results
            ],
        }
