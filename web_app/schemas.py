"""Pydantic models."""

from typing import List, Optional, Union

from pydantic import BaseModel


class ManifestCreate(BaseModel):
    """Create manifest Pydantic."""

    # Prompt params
    prompt: str
    n: int = 1
    max_tokens: int = 132
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    # Manifest client params
    client_name: str = "openai"
    client_connection: Optional[str] = None
    engine: str = "text-davinci-003"
    cache_name: str = "noop"
    cache_connection: Optional[str] = None


class ManifestResponse(BaseModel):
    """Manifest response Pydantic."""

    response: Union[str, List[str]]
    cached: bool
    request_params: dict
