"""Manifest class."""
import logging
from typing import Any, Iterable, List, Optional, Union

from manifest.response import Response

logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from manifest.caches.redis import RedisCache
from manifest.caches.sqlite import SQLiteCache
from manifest.clients.dummy import DummyClient
from manifest.clients.huggingface import HuggingFaceClient
from manifest.clients.openai import OpenAIClient
from manifest.prompt import Prompt

CLIENT_CONSTRUCTORS = {
    "openai": OpenAIClient,
    "huggingface": HuggingFaceClient,
    "dummy": DummyClient,
}

CACHE_CONSTRUCTORS = {
    "redis": RedisCache,
    "sqlite": SQLiteCache,
}


class Manifest:
    """Manifest session object."""

    def __init__(
        self,
        client_name: str = "openai",
        client_connection: Optional[str] = None,
        cache_name: str = "redis",
        cache_connection: str = "localhost:6379",
        stop_token: str = "",
        **kwargs: Any,
    ):
        """
        Initialize manifest.

        Args:
            client_name: name of client.
            client_connection: connection string for client.
            cache_name: name of cache.
            cache_connection: connection string for cache.
            stop_token: stop token prompt generation.
                        Can be overridden in run

        Remaining kwargs sent to client and cache.
        """
        if client_name not in CLIENT_CONSTRUCTORS:
            raise ValueError(
                f"Unknown client name: {client_name}. "
                f"Choices are {list(CLIENT_CONSTRUCTORS.keys())}"
            )
        if cache_name not in CACHE_CONSTRUCTORS:
            raise ValueError(
                f"Unknown cache name: {cache_name}. "
                f"Choices are {list(CACHE_CONSTRUCTORS.keys())}"
            )
        self.client_name = client_name
        self.client = CLIENT_CONSTRUCTORS[client_name](  # type: ignore
            client_connection, **kwargs
        )
        self.cache = CACHE_CONSTRUCTORS[cache_name](cache_connection, **kwargs)
        self.stop_token = stop_token

    def close(self) -> None:
        """Close the client and cache."""
        self.client.close()
        self.cache.close()

    def run(
        self,
        prompt: Prompt,
        input: Optional[Any] = None,
        overwrite_cache: bool = False,
        stop_token: Optional[str] = None,
        return_response: bool = False,
        **kwargs: Any,
    ) -> Union[str, List[str], Response]:
        """
        Run the prompt.

        Args:
            prompt: prompt to run.
            input: input to prompt.
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
                        Default is self.stop_token.
                        "" for no stop token.

        Returns:
            response from prompt.
        """
        stop_token = stop_token if stop_token is not None else self.stop_token
        prompt_str = prompt(input)
        possible_request, full_kwargs = self.client.get_request(prompt_str, **kwargs)
        # Create cacke key
        cache_key = full_kwargs.copy()
        # Make query model dependent
        cache_key["client_name"] = self.client_name
        # Make query prompt dependent
        cache_key["prompt"] = prompt_str
        response_obj = self.cache.get(cache_key, overwrite_cache, possible_request)
        # Extract text results
        if return_response:
            return response_obj
        else:
            return response_obj.get_response(stop_token)

    def run_batch(
        self,
        prompt: Prompt,
        input: Optional[Iterable[Any]] = None,
        overwrite_cache: bool = False,
        stop_token: Optional[str] = None,
        return_response: bool = False,
        **kwargs: Any,
    ) -> Iterable[Union[str, List[str], Response]]:
        """
        Run the prompt on a batch of inputs.

        Args:
            prompt: prompt to run.
            input: batch of inputs.
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
                        Default is self.stop_token.
                        "" for no stop token.

        Returns:
            batch of responses.
        """
        if input is None:
            input = [None]
        return [
            self.run(
                prompt, inp, overwrite_cache, stop_token, return_response, **kwargs
            )
            for inp in input
        ]

    def save_prompt(self, name: str, prompt: Prompt) -> None:
        """
        Save the prompt to the cache for long term storage.

        Args:
            name: name of prompt.
            prompt: prompt to save.
        """
        self.cache.set_key(name, prompt.serialize(), table="prompt")

    def load_prompt(self, name: str) -> Prompt:
        """
        Load the prompt from the cache.

        Args:
            name: name of prompt.

        Returns:
            Prompt saved with name.
        """
        return Prompt.deserialize(self.cache.get_key(name, table="prompt"))

    def open_explorer(self) -> None:
        """Open the explorer for jupyter widget."""
        # Open explorer
        # TODO: implement
        pass
