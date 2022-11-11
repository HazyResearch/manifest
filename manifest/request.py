"""Request object."""
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel


class Request(BaseModel):
    """Request object."""

    # Prompt
    prompt: Union[str, List[str]] = ""

    # Engine
    engine: str = "text-ada-001"

    # Temperature for generation
    temperature: float = 0.7

    # Max tokens for generation
    max_tokens: int = 100

    # Number completions
    n: int = 1

    # Nucleus sampling taking top_p probability mass tokens
    top_p: float = 1.0

    # Top k sampling taking top_k highest probability tokens
    top_k: int = 50

    # Stop sequences
    stop_sequences: Optional[List[str]] = None

    # Number beams beam search (HF)
    num_beams: int = 1

    # Whether to sample or do greedy (HF)
    do_sample: bool = False

    # Penalize repetition (HF)
    repetition_penalty: float = 1.0

    # Length penalty (HF)
    length_penalty: float = 1.0

    # Penalize resence
    presence_penalty: float = 0

    # Penalize frequency
    frequency_penalty: float = 0

    # Timeout
    client_timeout: int = 60

    def to_dict(
        self, allowable_keys: Dict[str, Tuple[str, Any]] = None, add_prompt: bool = True
    ) -> Dict[str, Any]:
        """
        Convert request to a dictionary.

        Add prompt ensures the prompt is always in the output dictionary.
        """
        if allowable_keys:
            include_keys = set(allowable_keys.keys())
            if add_prompt and "prompt":
                include_keys.add("prompt")
        else:
            allowable_keys = {}
            include_keys = None
        request_dict = {
            allowable_keys.get(k, (k, None))[0]: v
            for k, v in self.dict(include=include_keys).items()
            if v is not None
        }
        return request_dict
