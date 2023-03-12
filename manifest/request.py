"""Request object."""
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

NOT_CACHE_KEYS = {"client_timeout", "batch_size"}
DEFAULT_REQUEST_KEYS = {
    "client_timeout": ("client_timeout", 60),  # seconds
    "batch_size": ("batch_size", 1),
    "run_id": ("run_id", None),
    "request_type": ("request_type", None),
}


class Request(BaseModel):
    """Request object."""

    # Prompt
    prompt: Union[str, List[str]] = ""

    # Engine
    engine: str = "text-ada-001"

    # Number completions
    n: int = 1

    # Timeout
    client_timeout: int = 120

    # Run id used to repeat run with same parameters
    run_id: Optional[str] = None

    # Batch size for async batch run
    batch_size: int = 8

    # Request type None is for completion. Used to scoring prompt
    request_type: str = None

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


class LMRequest(Request):
    """Language Model Request object."""

    # Temperature for generation
    temperature: float = 0.7

    # Max tokens for generation
    max_tokens: int = 100

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


class DiffusionRequest(Request):
    """Diffusion Model Request object."""

    # Request type
    request_type: str = "diffusion"

    # Number of steps
    num_inference_steps: int = 50

    # Height of image
    height: int = 512

    # Width of image
    width: int = 512

    # Guidance scale
    guidance_scale: float = 7.5

    # Eta
    eta: float = 0.0
