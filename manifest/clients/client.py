"""Client class."""
import asyncio
import copy
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import aiohttp
import requests
from tenacity import RetryCallState, retry, stop_after_attempt, wait_random_exponential

from manifest.request import DEFAULT_REQUEST_KEYS, NOT_CACHE_KEYS, Request
from manifest.response import RESPONSE_CONSTRUCTORS, Response

logger = logging.getLogger(__name__)


def retry_if_ratelimit(retry_base: RetryCallState) -> bool:
    """Return whether to retry if ratelimited."""
    try:
        if isinstance(retry_base.outcome.exception(), requests.exceptions.HTTPError):
            exception = cast(
                requests.exceptions.HTTPError, retry_base.outcome.exception()
            )
            if exception.response.status_code == 429:  # type: ignore
                return True
    except Exception:
        pass
    return False


class Client(ABC):
    """Client class."""

    # Must be overridden by child class
    PARAMS: Dict[str, Tuple[str, Any]] = {}
    REQUEST_CLS = Request
    NAME: str = None

    def __init__(
        self, connection_str: Optional[str] = None, client_args: Dict[str, Any] = {}
    ):
        """
        Initialize client.

        kwargs are passed to client as default parameters.

        For clients like OpenAI that do not require a connection,
        the connection_str can be None.

        Args:
            connection_str: connection string for client.
            client_args: client arguments.
        """
        self.connect(connection_str, client_args)

    @abstractmethod
    def connect(
        self, connection_str: Optional[str], client_args: Dict[str, Any]
    ) -> None:
        """
        Connect to client.

        Args:
            connection_str: connection string.
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Close the client."""
        raise NotImplementedError()

    @abstractmethod
    def get_generation_url(self) -> str:
        """Get generation URL."""
        raise NotImplementedError()

    @abstractmethod
    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        raise NotImplementedError()

    @abstractmethod
    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference."""
        raise NotImplementedError()

    @abstractmethod
    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        raise NotImplementedError()

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(self.PARAMS.keys())

    def get_request(
        self, prompt: Union[str, List[str]], request_args: Dict[str, Any]
    ) -> Request:
        """
        Parse model kwargs to request.

        Args:
            prompt: prompt.
            request_args: request arguments.

        Returns:
            request.
        """
        params = {"prompt": prompt}
        for key in self.PARAMS:
            params[key] = request_args.pop(key, getattr(self, key))
        for key in DEFAULT_REQUEST_KEYS:
            if key not in params and key in request_args:
                params[key] = request_args.pop(key)
        return self.REQUEST_CLS(**params)  # type: ignore

    def get_request_params(self, request: Request) -> Dict[str, Any]:
        """Get request params.

        Add default keys that we need for requests such as batch_size.
        We drop these before sending to the model.
        """
        params_to_add = DEFAULT_REQUEST_KEYS.copy()
        params_to_add.update(self.PARAMS)
        request_params = request.to_dict(params_to_add)
        return request_params

    def get_cache_key(self, request: Request) -> Dict[str, Any]:
        """Get cache key for request.

        Skip keys that are not cache keys such as batch_size.
        """
        request_params = self.get_request_params(request)
        for key in NOT_CACHE_KEYS:
            request_params.pop(key, None)
        request_params.update(self.get_model_params())
        return request_params

    def split_usage(self, request: Dict, choices: List[str]) -> List[Dict[str, int]]:
        """Split usage into list of usages for each prompt."""
        return []

    def format_response(self, response: Dict, request: Dict) -> Dict[str, Any]:
        """
        Format response to dict.

        Args:
            response: response
            request: request

        Return:
            response as dict
        """
        if "choices" not in response:
            raise ValueError(f"Invalid response: {response}")
        if "usage" in response:
            # Handle splitting the usages for batch requests
            if len(response["choices"]) == 1:
                if isinstance(response["usage"], list):
                    response["usage"] = response["usage"][0]
                response["usage"] = [response["usage"]]
            else:
                # Try to split usage
                split_usage = self.split_usage(request, response["choices"])
                if split_usage:
                    response["usage"] = split_usage
        return response

    def split_requests(
        self, request_params: Dict[str, Any], batch_size: int, key: str = "prompt"
    ) -> List[Dict[str, Any]]:
        """Split request into batch_sized request.

        Args:
            request_params: request params.
            batch_size: batch size for requests.
            key: key to batch over

        Returns:
            list of request params.
        """
        data = copy.deepcopy(request_params[key])
        data_size = len(request_params[key])
        request_params_list = []
        for i in range(0, data_size, batch_size):
            params = copy.deepcopy(request_params)
            params[key] = data[i] if batch_size == 1 else data[i : i + batch_size]
            request_params_list.append(params)
        return request_params_list

    @retry(
        reraise=True,
        retry=retry_if_ratelimit,
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
    )
    def _run_completion(
        self, request_params: Dict[str, Any], retry_timeout: int
    ) -> Dict:
        """Execute completion request.

        Args:
            request_params: request params.
            retry_timeout: retry timeout.

        Returns:
            response as dict.
        """
        post_str = self.get_generation_url()
        res = requests.post(
            post_str,
            headers=self.get_generation_header(),
            json=request_params,
            timeout=retry_timeout,
        )
        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError:
            logger.error(res.json())
            raise requests.exceptions.HTTPError(res.json())
        return self.format_response(res.json(), request_params)

    @retry(
        reraise=True,
        retry=retry_if_ratelimit,
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
    )
    async def _arun_completion(
        self, request_params: Dict[str, Any], retry_timeout: int, batch_size: int
    ) -> Dict:
        """Async execute completion request.

        Args:
            request_params: request params.
            retry_timeout: retry timeout.
            batch_size: batch size for requests.

        Returns:
            response as dict.
        """
        post_str = self.get_generation_url()
        async with aiohttp.ClientSession(timeout=retry_timeout) as session:
            async with session.post(
                post_str,
                headers=self.get_generation_header(),
                json=request_params,
                timeout=retry_timeout,
            ) as res:
                res.raise_for_status()
                res_json = await res.json(content_type=None)
                return self.format_response(res_json, request_params)

    def run_request(self, request: Request) -> Response:
        """
        Get request string function.

        Args:
            request: request.

        Returns:
            response.
        """
        if isinstance(request.prompt, list) and not self.supports_batch_inference():
            raise ValueError(
                f"{self.__class__.__name__} does not support batch inference."
            )

        request_params = self.get_request_params(request)
        # Take the default keys we need and drop the rest as they
        # are not part of the model request.
        retry_timeout = request_params.pop("client_timeout")
        for key in DEFAULT_REQUEST_KEYS:
            request_params.pop(key, None)
        response_dict = self._run_completion(request_params, retry_timeout)
        return Response(
            response_dict,
            cached=False,
            request_params=request_params,
            **RESPONSE_CONSTRUCTORS.get(self.NAME, {}),  # type: ignore
        )

    async def arun_batch_request(self, request: Request) -> Response:
        """
        Get async request string function.

        Args:
            request: request.

        Returns:
            response.
        """
        required_batch_size = None
        if not self.supports_batch_inference():
            required_batch_size = 1
        if not isinstance(request.prompt, list):
            raise AssertionError(
                "request.prompt must be a list for async batch inference."
            )

        request_params = self.get_request_params(request)
        # Take the default keys we need and drop the rest as they
        # are not part of the model request.
        retry_timeout = request_params.pop("client_timeout")
        batch_size = request_params.pop("batch_size")
        batch_size = required_batch_size or batch_size
        for key in DEFAULT_REQUEST_KEYS:
            request_params.pop(key, None)

        num_batches = len(request.prompt) // batch_size
        if len(request.prompt) % batch_size != 0:
            batch_size = int(math.ceil(len(request.prompt) / (num_batches + 1)))

        request_batches = self.split_requests(request_params, batch_size)
        all_tasks = [
            asyncio.create_task(self._arun_completion(batch, retry_timeout, batch_size))
            for batch in request_batches
        ]
        responses = await asyncio.gather(*all_tasks)
        # Flatten responses
        choices = []
        usages = []
        for res_dict in responses:
            choices.extend(res_dict["choices"])
            if "usage" in res_dict:
                usages.extend(res_dict["usage"])
        final_response_dict = {"choices": choices}
        if usages:
            final_response_dict["usage"] = usages
        return Response(
            final_response_dict,
            cached=False,
            request_params=request_params,
            **RESPONSE_CONSTRUCTORS.get(self.NAME, {}),  # type: ignore
        )

    def get_score_prompt_request(
        self,
        request: Request,
    ) -> Response:
        """
        Get the logit score of the prompt via a forward pass of the model.

        Args:
            request: request.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support prompt scoring request."
        )
