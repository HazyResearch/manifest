"""Manifest class."""
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from manifest.caches.noop import NoopCache
from manifest.caches.postgres import PostgresCache
from manifest.caches.redis import RedisCache
from manifest.caches.sqlite import SQLiteCache
from manifest.clients.client import Client
from manifest.clients.huggingface import HuggingFaceClient
from manifest.connections.client_pool import (
    CLIENT_CONSTRUCTORS,
    ClientConnection,
    ClientConnectionPool,
)
from manifest.request import Request
from manifest.response import Response

logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


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
        client_name: Optional[str] = None,
        client_connection: Optional[str] = None,
        client_pool: Optional[List[ClientConnection]] = None,
        client_pool_schedule: str = "round_robin",
        cache_name: str = "noop",
        cache_connection: Optional[str] = None,
        stop_token: str = "",
        **kwargs: Any,
    ):
        """
        Initialize manifest.

        Args:
            client_name: name of client.
            client_connection: connection string for client.
            client_pool: list of client connections for multi-client.
            client_pool_schedule: schedule for client pool.
            cache_name: name of cache.
            cache_connection: connection string for cache.
            stop_token: stop token prompt generation.
                        Can be overridden in run

        Remaining kwargs sent to client and cache.
        """
        if not client_name and not client_pool:
            raise ValueError(
                "Must specify client_name or client_pool. "
                f"Choices are {list(CLIENT_CONSTRUCTORS.keys())}"
            )
        if client_name and client_pool:
            raise ValueError("Cannot specify both client_name and client_pool")
        if client_name:
            client_pool = [
                ClientConnection(
                    client_name=client_name,
                    client_connection=client_connection,
                    # Remove engine from kwargs
                    engine=kwargs.pop("engine", None),
                )
            ]
        self.client_pool = ClientConnectionPool(
            client_pool, client_pool_schedule, client_args=kwargs
        )
        if cache_name not in CACHE_CONSTRUCTORS:
            raise ValueError(
                f"Unknown cache name: {cache_name}. "
                f"Choices are {list(CACHE_CONSTRUCTORS.keys())}"
            )
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        self.cache = CACHE_CONSTRUCTORS[cache_name](  # type: ignore
            cache_connection, self.client_pool.request_type, cache_args=kwargs
        )
        if len(kwargs) > 0:
            raise ValueError(f"{list(kwargs.items())} arguments are not recognized.")

        self.stop_token = stop_token

    def close(self) -> None:
        """Close the client and cache."""
        self.client_pool.close()
        self.cache.close()

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

    def _split_cached_requests(
        self,
        request: Request,
        client: Client,
        overwrite_cache: bool,
    ) -> Tuple[Dict[int, Response], Request]:
        """Split a request into cached responses and Requests to run.

        Args:
            request: request object.
            overwrite_cache: whether to overwrite cache.

        Returns:
            cached_idx_to_response: dict of cached responses.
            new_request: request object with only prompts to run.
        """
        cached_idx_to_response: Dict[int, Response] = {}
        new_request = copy.deepcopy(request)
        if not overwrite_cache:
            if isinstance(new_request.prompt, list):
                new_request.prompt = []
                for idx, prompt_str in enumerate(request.prompt):
                    single_request = copy.deepcopy(request)
                    single_request.prompt = prompt_str
                    possible_response = self.cache.get(
                        client.get_cache_key(single_request)
                    )
                    if possible_response:
                        cached_idx_to_response[idx] = possible_response
                    else:
                        new_request.prompt.append(prompt_str)
            else:
                possible_response = self.cache.get(client.get_cache_key(new_request))
                if possible_response:
                    cached_idx_to_response[0] = possible_response
                    new_request.prompt = None
        return cached_idx_to_response, new_request

    def _stitch_responses_and_cache(
        self,
        request: Request,
        client: Client,
        response: Union[Response, None],
        cached_idx_to_response: Dict[int, Response],
    ) -> Response:
        """Stich together the cached and uncached responses."""
        # We stitch the responses (the choices) here from both the new request the
        # cached entries.
        all_model_choices = []
        all_usages = []
        all_input_prompts = []
        response_idx = 0
        number_prompts = len(cached_idx_to_response)
        single_output = False
        if response:
            if isinstance(response.get_request()["prompt"], str):
                single_output = True
                number_prompts += 1
            else:
                number_prompts += len(response.get_request()["prompt"])
        response_gen_key = None
        response_logits_key = None
        response_item_key = None
        for idx in range(number_prompts):
            if idx in cached_idx_to_response:
                cached_res = cached_idx_to_response[idx]
                response_gen_key = cached_res.generation_key
                response_logits_key = cached_res.logits_key
                response_item_key = cached_res.item_key
                response_usage_key = cached_res.usage_key
                all_input_prompts.append(cached_res.get_request()["prompt"])
                json_response = cached_res.get_json_response()
                if request.n == 1:
                    assert (
                        len(json_response[response_gen_key]) == 1
                    ), "cached response should have only one choice"
                all_model_choices.extend(json_response[response_gen_key])
                if response_usage_key:
                    all_usages.extend(json_response[response_usage_key])
            else:
                assert response is not None, "response should not be None"
                response = cast(Response, response)
                response_gen_key = response.generation_key
                response_logits_key = response.logits_key
                response_item_key = response.item_key
                response_usage_key = response.usage_key
                # the choices list in the response is a flat one.
                # length is request.n * num_prompts
                current_choices = response.get_json_response()[response_gen_key][
                    response_idx * request.n : (response_idx + 1) * request.n
                ]
                all_model_choices.extend(current_choices)

                if isinstance(response.get_request()["prompt"], list):
                    prompt = response.get_request()["prompt"][response_idx]
                else:
                    prompt = str(response.get_request()["prompt"])
                if response_usage_key:
                    usage = response.get_json_response()[response_usage_key][
                        response_idx * request.n : (response_idx + 1) * request.n
                    ]
                    all_usages.extend(usage)
                all_input_prompts.append(prompt)
                # set cache
                new_request = copy.deepcopy(request)
                new_request.prompt = prompt
                cache_key = client.get_cache_key(new_request)
                new_response_key = copy.deepcopy(response.get_json_response())
                new_response_key[response_gen_key] = current_choices
                if response_usage_key:
                    new_response_key[response_usage_key] = usage
                self.cache.set(cache_key, new_response_key)
                response_idx += 1

        new_request = copy.deepcopy(request)
        new_request.prompt = (
            all_input_prompts
            if len(all_input_prompts) > 1 or not single_output
            else all_input_prompts[0]
        )
        new_response = {response_gen_key: all_model_choices}
        if response_usage_key:
            new_response[response_usage_key] = all_usages
        response_obj = Response(
            new_response,
            cached=len(cached_idx_to_response) > 0,
            request_params=client.get_cache_key(new_request),
            generation_key=response_gen_key,
            logits_key=response_logits_key,
            item_key=response_item_key,
            usage_key=response_usage_key,
        )
        return response_obj

    def run(
        self,
        prompt: Union[str, List[str]],
        overwrite_cache: bool = False,
        stop_token: Optional[str] = None,
        return_response: bool = False,
        **kwargs: Any,
    ) -> Union[str, List[str], np.ndarray, List[np.ndarray], Response]:
        """
        Run the prompt.

        Args:
            prompt: prompt(s) to run.
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
                        Default is self.stop_token.
                        "" for no stop token.
            return_response: whether to return Response object.

        Returns:
            response from prompt.
        """
        is_batch = isinstance(prompt, list)
        # Get the client to run
        client = self.client_pool.get_client()
        stop_token = stop_token if stop_token is not None else self.stop_token
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = client.get_request(prompt, kwargs)
        # Avoid nested list of results - enforce n = 1 for batch
        if is_batch and request_params.n > 1:
            raise ValueError("Batch mode does not support n > 1.")
        self._validate_kwargs(kwargs, request_params)

        cached_idx_to_response, request_params = self._split_cached_requests(
            request_params, client, overwrite_cache
        )
        # If not None value or empty list - run new request
        if request_params.prompt:
            # Start timing metrics
            self.client_pool.start_timer()
            response = client.run_request(request_params)
            self.client_pool.end_timer()
        else:
            # Nothing to run
            response = None

        final_response = self._stitch_responses_and_cache(
            request=request_params,
            client=client,
            response=response,
            cached_idx_to_response=cached_idx_to_response,
        )

        # Extract text results
        if return_response:
            return final_response
        else:
            return final_response.get_response(stop_token, is_batch)

    async def arun_batch(
        self,
        prompts: List[str],
        overwrite_cache: bool = False,
        stop_token: Optional[str] = None,
        return_response: bool = False,
        **kwargs: Any,
    ) -> Union[List[str], List[np.ndarray], Response]:
        """
        Run a batch of prompts with async.

        Args:
            prompts: prompts to run.
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
                        Default is self.stop_token.
                        "" for no stop token.
            return_response: whether to return Response object.

        Returns:
            response from prompt.
        """
        # Get the client to run
        client = self.client_pool.get_client()
        stop_token = stop_token if stop_token is not None else self.stop_token
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = client.get_request(prompts, kwargs)
        # Avoid nested list of results - enforce n = 1 for batch
        if request_params.n > 1:
            raise ValueError("Batch mode does not support n > 1.")
        self._validate_kwargs(kwargs, request_params)

        cached_idx_to_response, request_params = self._split_cached_requests(
            request_params, client, overwrite_cache
        )
        # If not None value or empty list - run new request
        if request_params.prompt:
            self.client_pool.start_timer()
            response = await client.arun_batch_request(request_params)
            self.client_pool.end_timer()
        else:
            # Nothing to run
            response = None

        final_response = self._stitch_responses_and_cache(
            request=request_params,
            client=client,
            response=response,
            cached_idx_to_response=cached_idx_to_response,
        )

        # Extract text results
        if return_response:
            return final_response
        else:
            return cast(
                Union[List[str], List[np.ndarray]],
                final_response.get_response(stop_token, True),
            )

    def score_prompt(
        self,
        prompt: Union[str, List[str]],
        overwrite_cache: bool = False,
        **kwargs: Any,
    ) -> Dict:
        """
        Score the prompt via forward pass of the model - no sampling or generation.

        Returns the response object with logits of the prompt.

        Args:
            prompt: prompt(s) to run.
            overwrite_cache: whether to overwrite cache.

        Returns:
            response from prompt.
        """
        client = self.client_pool.get_client()
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = client.get_request(prompt, kwargs)
        request_params.request_type = "score_prompt"

        if request_params.n > 1:
            raise ValueError("Sequence scoring does not support n > 1.")
        self._validate_kwargs(kwargs, request_params)

        cached_idx_to_response, request_params = self._split_cached_requests(
            request_params, client, overwrite_cache
        )
        # If not None value or empty list - run new request
        if request_params.prompt:
            try:
                response = cast(HuggingFaceClient, client).get_score_prompt_request(
                    request_params
                )
            except AttributeError:
                raise ValueError("`score_prompt` only supported for HF models.")
        else:
            # Nothing to run
            response = None

        final_response = self._stitch_responses_and_cache(
            request=request_params,
            client=client,
            response=response,
            cached_idx_to_response=cached_idx_to_response,
        )
        return final_response.to_dict()
