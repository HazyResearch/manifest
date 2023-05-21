"""Client class."""
import asyncio
import copy
import json
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast

import aiohttp
import requests
import tqdm.asyncio
from tenacity import RetryCallState, retry, stop_after_attempt, wait_random_exponential

from manifest.request import (
    DEFAULT_REQUEST_KEYS,
    NOT_CACHE_KEYS,
    LMChatRequest,
    LMRequest,
    LMScoreRequest,
    Request,
)
from manifest.response import (
    RESPONSE_CONSTRUCTORS,
    ArrayModelChoice,
    LMModelChoice,
    ModelChoices,
    Response,
    Usage,
    Usages,
)

logger = logging.getLogger(__name__)

ATTEMPTS_BEFORE_STOP = 20
ATTEMPTS_TIMEOUT = 120
# http_status mainly for azure and e.code mainly for openai usage
# e.http_status == 408 occurs when Azure times out
# e.code == 429 rate lime
# e.code == 500 or 502 occurs when server error
API_ERROR_CODE = {408, 429, 500, 502}


def retry_if_ratelimit(retry_base: RetryCallState) -> bool:
    """Return whether to retry if ratelimited."""
    try:
        if isinstance(retry_base.outcome.exception(), requests.exceptions.HTTPError):
            exception = cast(
                requests.exceptions.HTTPError, retry_base.outcome.exception()
            )
            # 500 is a server error, 429 is a rate limit error
            if exception.response.status_code in API_ERROR_CODE:  # type: ignore
                return True
    except Exception:
        pass
    return False


def return_error_response(retry_state: RetryCallState) -> dict:
    """Return error response if all retries failed."""
    request_params = retry_state.args[1]
    number_of_prompts = (
        len(request_params["prompt"])
        if "prompt" in request_params
        else len(request_params["messages"])
    )
    return {
        "choices": [],
        "usage": {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
        "errors": [str(retry_state.outcome.exception())] * number_of_prompts,
    }


class Client(ABC):
    """Client class."""

    # Must be overridden by child class
    PARAMS: Dict[str, Tuple[str, Any]] = {}
    REQUEST_CLS = Request
    NAME: str = None
    IS_CHAT: bool = False

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

        Override in child client class.
        Args:
            connection_str: connection string.
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Close the client.

        Override in child client class.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_generation_url(self) -> str:
        """Get generation URL.

        Override in child client class.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Override in child client class.
        Returns:
            header.
        """
        raise NotImplementedError()

    @abstractmethod
    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference.

        Override in child client class.
        """
        raise NotImplementedError()

    @abstractmethod
    def supports_streaming_inference(self) -> bool:
        """Return whether the client supports streaming inference.

        Override in child client class.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Override in child client class.
        Returns:
            model params.
        """
        raise NotImplementedError()

    def get_tokenizer(self, model: str) -> Tuple[Any, int]:
        """Get tokenizer for model.

        Override in child client class. Return None, -1 if not supported
        or no prompt truncation required.
        Returns:
            tokenizer: tokenizer with encoder and decode
            max_length: max length of model
        """
        return None, -1

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(self.PARAMS.keys())

    def split_usage(self, request: Dict, choices: List[str]) -> List[Dict[str, int]]:
        """Split usage into list of usages for each prompt."""
        # TODO: add this in using default tokenizer
        return []

    def preprocess_request_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess request params.

        Override in child client class to reformat requests to model.

        Args:
            request: request params.

        Returns:
            request params.
        """
        return request

    def postprocess_response(
        self, response: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Postprocess and validate response as dict.

        Override in child client class to reform model responses.

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
        # Adds default values from self.PARAMS if not in request_args
        for key in self.PARAMS:
            params[key] = request_args.pop(key, getattr(self, key))
        # Allows for overriding DEFAULT_REQUEST_KEYS even if they are not
        # in self.PARAMS. Note that DEFAULT_REQUEST_KEYS match the default
        # values in Request.
        for key in DEFAULT_REQUEST_KEYS:
            if key not in params and key in request_args:
                params[key] = request_args.pop(key)
        return self.REQUEST_CLS(**params)  # type: ignore

    def _get_request_params(self, request: Request) -> Dict[str, Any]:
        """Get request params.

        Add default keys that we need for requests such as batch_size.
        We drop these before sending to the model.
        """
        params_to_add = DEFAULT_REQUEST_KEYS.copy()
        # This will override DEFAULT_REQUEST_KEYS with those in PARAMS
        params_to_add.update(self.PARAMS)
        # to_dict will handle parameter renaming but not any
        # default value handling - that is done in get_request()
        request_params = request.to_dict(params_to_add)
        return request_params

    def get_cache_key(self, request: Request) -> Dict[str, Any]:
        """Get cache key for request.

        Skip keys that are not cache keys such as batch_size.
        """
        request_params = self._get_request_params(request)
        for key in NOT_CACHE_KEYS:
            request_params.pop(key, None)
        # Make sure to add model params and request class
        request_params.update(self.get_model_params())
        request_params["request_cls"] = request.__class__.__name__
        return request_params

    def _split_requests(
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

    def _get_model_choices(self, response: Dict) -> ModelChoices:
        """Format response to ModelChoices."""
        # Array or text response
        response_type = RESPONSE_CONSTRUCTORS[self.REQUEST_CLS]["response_type"]
        if response_type == "array":
            choices: List[Union[LMModelChoice, ArrayModelChoice]] = [
                ArrayModelChoice(**choice) for choice in response["choices"]
            ]
        else:
            choices = [LMModelChoice(**choice) for choice in response["choices"]]
        return ModelChoices(choices=choices)

    def _stitch_responses(self, request: Request, responses: List[Dict]) -> Response:
        """Stitch responses together.

        Useful for batch requests.
        """
        choices = []
        usages = []
        for res_dict in responses:
            choices.extend(res_dict["choices"])
            if "usage" in res_dict:
                usages.extend(res_dict["usage"])
        final_response_dict = {"choices": choices}
        final_usages = None
        if usages:
            final_usages = Usages(usages=[Usage(**usage) for usage in usages])
        # TODO: Add usage based on tokenizer
        return Response(
            self._get_model_choices(final_response_dict),
            cached=False,
            request=request,
            usages=final_usages,
            **RESPONSE_CONSTRUCTORS[self.REQUEST_CLS],  # type: ignore
        )

    def _verify_request_lengths(
        self, request: Dict[str, Any], model: str, max_tokens: int
    ) -> None:
        """Verify that the request length is not too long."""
        encoder, max_length = self.get_tokenizer(model)
        if not encoder or max_length < 0:
            return
        if isinstance(request["prompt"], str):
            prompts = [request["prompt"]]
        else:
            prompts = request["prompt"]
        for i in range(len(prompts)):
            prompt = prompts[i]
            encoded_prompt = encoder.encode(prompt)
            if len(encoded_prompt) + max_tokens > max_length:
                logger.warning(
                    f"Prompt {prompt} is too long for model {model}. "
                    "Truncating prompt from left."
                )
                # -20 to be safe
                prompt = encoder.decode(
                    encoded_prompt[-int(max_length - max_tokens - 20) :]
                )
                prompts[i] = prompt
        if isinstance(request["prompt"], str):
            request["prompt"] = prompts[0]
        else:
            request["prompt"] = prompts

    @retry(
        reraise=True,
        retry=retry_if_ratelimit,
        wait=wait_random_exponential(min=1, max=ATTEMPTS_TIMEOUT),
        stop=stop_after_attempt(ATTEMPTS_BEFORE_STOP),
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
        request_params = self.preprocess_request_params(request_params)
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
        return self.postprocess_response(res.json(), request_params)

    @retry(
        reraise=True,
        retry=retry_if_ratelimit,
        wait=wait_random_exponential(min=1, max=ATTEMPTS_TIMEOUT),
        stop=stop_after_attempt(ATTEMPTS_BEFORE_STOP),
    )
    async def _arun_completion(
        self, request_params: Dict[str, Any], retry_timeout: int
    ) -> Dict:
        """Async execute completion request.

        Args:
            request_params: request params.
            retry_timeout: retry timeout.

        Returns:
            response as dict.
        """
        request_params = self.preprocess_request_params(request_params)
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
                return self.postprocess_response(res_json, request_params)

    @retry(
        reraise=True,
        retry=retry_if_ratelimit,
        wait=wait_random_exponential(min=1, max=ATTEMPTS_TIMEOUT),
        stop=stop_after_attempt(ATTEMPTS_BEFORE_STOP),
    )
    def _run_streaming_completion(
        self, request_params: Dict[str, Any], retry_timeout: int
    ) -> Generator[Dict, None, None]:
        """Execute completion request streaming.

        Args:
            request_params: request params.
            retry_timeout: retry timeout.

        Returns:
            response as dict.
        """
        request_params = self.preprocess_request_params(request_params)
        request_params["stream"] = True
        post_str = self.get_generation_url()
        res_iter = requests.post(
            post_str,
            headers=self.get_generation_header(),
            json=request_params,
            timeout=retry_timeout,
            stream=True,
        )
        for res_token in res_iter.iter_lines():
            if res_token:
                decoded_res_token = res_token.decode("utf-8")
                decoded_res_token = decoded_res_token.replace("data: ", "")
                if decoded_res_token == "[DONE]":
                    break
                try:
                    decoded_res_token_dct = json.loads(decoded_res_token)
                    postprocess_res_token_dct = self.postprocess_response(
                        decoded_res_token_dct, request_params
                    )
                    # If nothing is returned, skip
                    if (
                        not postprocess_res_token_dct
                        or not postprocess_res_token_dct["choices"]
                    ):
                        continue
                    yield postprocess_res_token_dct
                except Exception as e:
                    raise e

    def run_request(self, request: Request) -> Response:
        """
        Run request.

        Args:
            request: request.

        Returns:
            response.
        """
        # Make everything list for consistency
        if isinstance(request.prompt, list):
            prompt_list = request.prompt
        else:
            prompt_list = [request.prompt]

        request_params = self._get_request_params(request)
        # Set the params as a list. Do not set the request
        # object itself as the cache will then store it as a
        # list which is inconsistent with the request input.
        request_params["prompt"] = prompt_list

        # If batch_size is not set, set it to 1
        batch_size = request_params.pop("batch_size") or 1
        if not self.supports_batch_inference() and batch_size != 1:
            logger.warning(
                f"{self.__class__.__name__} does not support batch inference."
                " Setting batch size to 1"
            )
            batch_size = 1

        # Take the default keys we need and drop the rest as they
        # are not part of the model request.
        retry_timeout = request_params.pop("client_timeout")
        for key in DEFAULT_REQUEST_KEYS:
            request_params.pop(key, None)

        # Make sure requests are in the request length
        # If no tokenizer is set or not LM request, this
        # will do nothing
        if isinstance(request, LMRequest):
            self._verify_request_lengths(
                request_params, model=request.engine, max_tokens=request.max_tokens
            )

        # Batch requests
        num_batches = len(prompt_list) // batch_size
        if len(prompt_list) % batch_size != 0:
            batch_size = int(math.ceil(len(prompt_list) / (num_batches + 1)))
        request_batches = self._split_requests(request_params, batch_size)

        response_dicts = [
            self._run_completion(batch, retry_timeout) for batch in request_batches
        ]
        # Flatten responses
        return self._stitch_responses(request, response_dicts)

    async def arun_batch_request(
        self, request: Request, verbose: bool = False
    ) -> Response:
        """
        Run async request.

        Args:
            request: request.s

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

        request_params = self._get_request_params(request)
        # Take the default keys we need and drop the rest as they
        # are not part of the model request.
        retry_timeout = request_params.pop("client_timeout")
        batch_size = request_params.pop("batch_size")
        batch_size = required_batch_size or batch_size
        for key in DEFAULT_REQUEST_KEYS:
            request_params.pop(key, None)

        # Make sure requests are in the request length
        # If no tokenizer is set or not LM request, this
        # will do nothing
        if isinstance(request, LMRequest):
            self._verify_request_lengths(
                request_params, model=request.engine, max_tokens=request.max_tokens
            )

        # Batch requests
        num_batches = len(request.prompt) // batch_size
        if len(request.prompt) % batch_size != 0:
            batch_size = int(math.ceil(len(request.prompt) / (num_batches + 1)))

        request_batches = self._split_requests(request_params, batch_size)
        all_tasks = [
            asyncio.create_task(self._arun_completion(batch, retry_timeout))
            for batch in request_batches
        ]
        responses = await tqdm.asyncio.tqdm.gather(*all_tasks, disable=not verbose)
        # Flatten responses
        return self._stitch_responses(request, responses)

    def run_chat_request(
        self,
        request: LMChatRequest,
    ) -> Response:
        """
        Get the response from chat model.

        Args:
            request: request.

        Returns:
            response.
        """
        request_params = self._get_request_params(request)
        # Take the default keys we need and drop the rest as they
        # are not part of the model request.
        retry_timeout = request_params.pop("client_timeout")
        for key in DEFAULT_REQUEST_KEYS:
            request_params.pop(key, None)

        # Make sure requests are in the request length
        # If no tokenizer is set or not LM request, this
        # will do nothing
        self._verify_request_lengths(
            request_params, model=request.engine, max_tokens=request.max_tokens
        )

        response_dict = self._run_completion(request_params, retry_timeout)
        usages = None
        if "usage" in response_dict:
            usages = [Usage(**usage) for usage in response_dict["usage"]]

        return Response(
            response=self._get_model_choices(response_dict),
            cached=False,
            request=request,
            usages=Usages(usages=usages) if usages else None,
            **RESPONSE_CONSTRUCTORS[LMChatRequest],  # type: ignore
        )

    def run_streaming_request(
        self, request: Request
    ) -> Generator[Response, None, None]:
        """
        Run streaming request.

        Args:
            request: request.

        Returns:
            response.
        """
        if not isinstance(request.prompt, str):
            raise ValueError("Streaming requests must have a single prompt.")
        if not self.supports_streaming_inference():
            raise ValueError(
                f"{self.__class__.__name__} does not support streaming inference."
            )
        request_params = self._get_request_params(request)

        # Take the default keys we need and drop the rest as they
        # are not part of the model request.
        retry_timeout = request_params.pop("client_timeout")
        for key in DEFAULT_REQUEST_KEYS:
            request_params.pop(key, None)

        # Make sure requests are in the request length
        # If no tokenizer is set or not LM request, this
        # will do nothing
        if isinstance(request, LMRequest):
            self._verify_request_lengths(
                request_params, model=request.engine, max_tokens=request.max_tokens
            )

        for token_response in self._run_streaming_completion(
            request_params, retry_timeout
        ):
            yield self._stitch_responses(request, [token_response])

    def run_score_prompt_request(
        self,
        request: LMScoreRequest,
    ) -> Response:
        """
        Get the logit score of the prompt via a forward pass of the model.

        Args:
            request: request.

        Returns:
            response.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support prompt scoring request."
        )
