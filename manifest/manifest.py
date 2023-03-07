"""Manifest class."""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from manifest.caches.noop import NoopCache
from manifest.caches.postgres import PostgresCache
from manifest.caches.redis import RedisCache
from manifest.caches.sqlite import SQLiteCache
from manifest.clients.ai21 import AI21Client
from manifest.clients.cohere import CohereClient
from manifest.clients.dummy import DummyClient
from manifest.clients.huggingface import HuggingFaceClient
from manifest.clients.openai import OpenAIClient
from manifest.clients.toma import TOMAClient
from manifest.request import Request
from manifest.response import Response
from manifest.session import Session

logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

CLIENT_CONSTRUCTORS = {
    "openai": OpenAIClient,
    "cohere": CohereClient,
    "ai21": AI21Client,
    "huggingface": HuggingFaceClient,
    "dummy": DummyClient,
    "toma": TOMAClient,
}

# ChatGPT
try:
    from manifest.clients.chatgpt import ChatGPTClient

    CLIENT_CONSTRUCTORS["chatgpt"] = ChatGPTClient
except Exception:
    logger.info("ChatGPT not installed. Skipping import.")
    pass

# Diffusion
DIFFUSION_CLIENTS = ["diffuser", "tomadiffuser"]
try:
    from manifest.clients.diffuser import DiffuserClient
    from manifest.clients.toma_diffuser import TOMADiffuserClient

    CLIENT_CONSTRUCTORS["diffuser"] = DiffuserClient
    CLIENT_CONSTRUCTORS["tomadiffuser"] = TOMADiffuserClient
except Exception:
    logger.info("Diffusion not supported. Skipping import.")
    pass


CACHE_CONSTRUCTORS = {
    "redis": RedisCache,
    "sqlite": SQLiteCache,
    "noop": NoopCache,
    "postgres": PostgresCache,
}


class Manifest:
    """Manifest session object."""

    def __init__(
        self,
        client_name: str = "openai",
        client_connection: Optional[str] = None,
        cache_name: str = "noop",
        cache_connection: Optional[str] = None,
        session_id: Optional[str] = None,
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
            session_id: session id for user session cache.
                        None (default) means no session logging.
                        "_default" means generate new session id.
            stop_token: stop token prompt generation.
                        Can be overridden in run

        Remaining kwargs sent to client and cache.
        """
        if client_name not in CLIENT_CONSTRUCTORS:
            if client_name in DIFFUSION_CLIENTS:
                raise ImportError(
                    f"Diffusion client {client_name} requires the proper install. "
                    "Make sure to run `pip install manifest-ml[diffusers]` "
                    "or install Pillow."
                )
            else:
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
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        self.cache = CACHE_CONSTRUCTORS[cache_name](  # type: ignore
            cache_connection, self.client_name, cache_args=kwargs
        )
        self.client = CLIENT_CONSTRUCTORS[self.client_name](  # type: ignore
            client_connection, client_args=kwargs
        )
        if session_id is not None:
            if self.client_name == "diffuser":
                raise NotImplementedError(
                    "Session logging not implemented for Diffuser client."
                )
            if session_id == "_default":
                # Set session_id to None for Session random id
                session_id = None
            self.session = Session(session_id)
        else:
            self.session = None
        if len(kwargs) > 0:
            raise ValueError(f"{list(kwargs.items())} arguments are not recognized.")

        self.stop_token = stop_token

    def close(self) -> None:
        """Close the client and cache."""
        self.client.close()
        self.cache.close()

    def change_client(
        self,
        client_name: Optional[str] = None,
        client_connection: Optional[str] = None,
        stop_token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Change manifest client.

        Args:
            client_name: name of client.
            client_connection: connection string for client.
            stop_token: stop token prompt generation.
                        Can be overridden in run

        Remaining kwargs sent to client.
        """
        if client_name:
            if client_name not in CLIENT_CONSTRUCTORS:
                raise ValueError(
                    f"Unknown client name: {client_name}. "
                    f"Choices are {list(CLIENT_CONSTRUCTORS.keys())}"
                )
            self.client_name = client_name
            self.client = CLIENT_CONSTRUCTORS[client_name](  # type: ignore
                client_connection, client_args=kwargs
            )
            if len(kwargs) > 0:
                raise ValueError(
                    f"{list(kwargs.items())} arguments are not recognized."
                )

        if stop_token is not None:
            self.stop_token = stop_token

    def _validate_kwargs(self, kwargs: Dict, request_params: Request) -> None:
        """Validate kwargs.

        Args:
            kwargs: kwargs to validate.
            request_params: request object to validate against.
        """
        # Check for invalid kwargs
        non_request_kwargs = [
            (k, v) for k, v in kwargs.items() if k not in request_params.__dict__
        ]
        if len(non_request_kwargs) > 0:
            raise ValueError(
                f"{list(non_request_kwargs)} arguments are not recognized."
            )

        # Warn for valid but unused kwargs
        request_unused_kwargs = [
            (k, v) for k, v in kwargs.items() if k not in non_request_kwargs
        ]
        if len(request_unused_kwargs) > 0:
            logger.warning(f"{list(request_unused_kwargs)} arguments are unused.")
        return

    def run(
        self,
        prompt: Union[str, List[str]],
        overwrite_cache: bool = False,
        run_id: Optional[str] = None,
        stop_token: Optional[str] = None,
        return_response: bool = False,
        **kwargs: Any,
    ) -> Union[str, List[str], np.ndarray, List[np.ndarray], Response]:
        """
        Run the prompt.

        Args:
            prompt: prompt(s) to run.
            overwrite_cache: whether to overwrite cache.
            run_id: run id for cache to repeat same run.
            stop_token: stop token for prompt generation.
                        Default is self.stop_token.
                        "" for no stop token.
            return_response: whether to return Response object.

        Returns:
            response from prompt.
        """
        is_batch = isinstance(prompt, list)

        stop_token = stop_token if stop_token is not None else self.stop_token
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = self.client.get_request_params(prompt, kwargs)
        # Avoid nested list of results - enforce n = 1 for batch
        if is_batch and request_params.n > 1:
            raise ValueError("Batch mode does not support n > 1.")
        possible_request, full_kwargs = self.client.get_request(request_params)

        self._validate_kwargs(kwargs, request_params)
        
        # Create cacke key
        cache_key = full_kwargs.copy()
        # Make query model dependent
        cache_key.update(self.client.get_model_params())
        if run_id:
            cache_key["run_id"] = run_id
        response_obj = self.cache.get(cache_key, overwrite_cache, possible_request)
        # Log session dictionary values
        if self.session:
            self.session.log_query(cache_key, response_obj.to_dict())
        # Extract text results
        if return_response:
            return response_obj
        else:
            return response_obj.get_response(stop_token, is_batch)

    def score_prompt(
        self,
        prompt: Union[str, List[str]],
        overwrite_cache: bool = False,
        **kwargs: Any,
    ) -> Dict:
        """
        Score the prompt via forward pass of the model - no sampling or generation.

        Returns the response object with logits of the prompt.

        Prompt scoring is not part of a session cache.

        Args:
            prompt: prompt(s) to run.
            overwrite_cache: whether to overwrite cache.

        Returns:
            response from prompt.
        """
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = self.client.get_request_params(prompt, kwargs)

        if request_params.n > 1:
            raise ValueError("Sequence scoring does not support n > 1.")

        try:
            possible_request, full_kwargs = cast(
                HuggingFaceClient, self.client
            ).get_score_prompt_request(request_params)
        except AttributeError:
            raise ValueError("`score_prompt` only supported for HF models.")

        self._validate_kwargs(kwargs, request_params)
        # Create cacke key
        cache_key = full_kwargs.copy()
        # Make query model dependent
        cache_key.update(self.client.get_model_params())
        response_obj = self.cache.get(cache_key, overwrite_cache, possible_request)
        return response_obj.to_dict()

    def get_last_queries(
        self,
        last_n: int = -1,
        return_raw_values: bool = False,
        stop_token: Optional[str] = None,
    ) -> List[Tuple[Any, Any]]:
        """
        Get last n queries from current session.

        If last_n is -1, return all queries. By default will only return the
        prompt text and result text unles return_raw_values is False.

        Args:
            last_n: last n queries.
            return_raw_values: whether to return raw values as dicts.
            stop_token: stop token for prompt results to be applied to all results.

        Returns:
            last n list of queries and outputs.
        """
        if self.session is None:
            raise ValueError(
                "Session was not initialized. Set `session_id` when loading Manifest."
            )
        stop_token = stop_token if stop_token is not None else self.stop_token
        last_queries = self.session.get_last_queries(last_n)
        if not return_raw_values:
            last_queries = [
                (
                    query["prompt"],
                    Response.from_dict(response).get_response(
                        stop_token, is_batch=isinstance(query["prompt"], list)
                    ),
                )  # type: ignore
                for query, response in last_queries
            ]
        return last_queries

    def open_explorer(self) -> None:
        """Open the explorer for jupyter widget."""
        # Open explorer
        # TODO: implement
        pass
