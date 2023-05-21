"""Client response."""
import copy
import json
from typing import Any, Dict, Generator, List, Optional, Type, Union, cast

import numpy as np
from pydantic import BaseModel

from manifest.request import (
    ENGINE_SEP,
    DiffusionRequest,
    EmbeddingRequest,
    LMChatRequest,
    LMRequest,
    LMScoreRequest,
    Request,
)

RESPONSE_CONSTRUCTORS: Dict[Type[Request], Dict[str, Union[str, Type[Request]]]] = {
    LMRequest: {"response_type": "text", "request_type": LMRequest},
    LMChatRequest: {"response_type": "text", "request_type": LMChatRequest},
    LMScoreRequest: {"response_type": "text", "request_type": LMScoreRequest},
    EmbeddingRequest: {"response_type": "array", "request_type": EmbeddingRequest},
    DiffusionRequest: {"response_type": "array", "request_type": DiffusionRequest},
}


class NumpyArrayEncoder(json.JSONEncoder):
    """Numpy array encoder."""

    def default(self, obj: Any) -> str:
        """Encode numpy array."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Usage(BaseModel):
    """Prompt usage class."""

    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0


class Usages(BaseModel):
    """Prompt usage class."""

    usages: List[Usage]


class LMModelChoice(BaseModel):
    """Model single completion."""

    text: str
    token_logprobs: Optional[List[float]] = None
    tokens: Optional[List[str]] = None


class ArrayModelChoice(BaseModel):
    """Model single completion."""

    array: np.ndarray
    token_logprobs: Optional[List[float]] = None

    class Config:
        """Pydantic config class."""

        arbitrary_types_allowed = True


class ModelChoices(BaseModel):
    """Model choices."""

    choices: List[Union[LMModelChoice, ArrayModelChoice]]


class Response:
    """Response class."""

    def __init__(
        self,
        response: ModelChoices,
        cached: bool,
        request: Request,
        response_type: str,
        request_type: Type[Request],
        usages: Optional[Usages] = None,
    ):
        """
        Initialize response.

        Args:
            response: response dict.
            usages: usage dict.
            cached: whether response is cached.
            request: request.
            response_type: response type.
            request_type: request type.
        """
        self._item_dtype = None
        self._response_type = response_type
        if self._response_type not in {"array", "text"}:
            raise ValueError(f"Invalid response type {self._response_type}")
        self._request_type = request_type
        self._response = response
        self._usages = usages or Usages(usages=[])
        self._cached = cached
        self._request = request
        if self._response.choices:
            if response_type == "array":
                if not isinstance(self._response.choices[0], ArrayModelChoice):
                    raise ValueError(
                        "response_type is array but response is "
                        f"{self._response.choices[0].__class__}"
                    )
                self._item_dtype = str(
                    cast(ArrayModelChoice, self._response.choices[0]).array.dtype
                )
            else:
                if not isinstance(self._response.choices[0], LMModelChoice):
                    raise ValueError(
                        "response_type is text but response is "
                        f"{self._response.choices[0].__class__}"
                    )

    def is_cached(self) -> bool:
        """Check if response is cached."""
        return self._cached

    def get_request_obj(self) -> Request:
        """Get request parameters."""
        return self._request

    def get_response_obj(self) -> ModelChoices:
        """Get response object."""
        return self._response

    def get_usage_obj(self) -> Usages:
        """Get usage object."""
        return self._usages

    def get_json_response(self) -> Dict:
        """Get response dict without parsing."""
        return self._response.dict()

    def get_response(
        self, stop_token: str = "", is_batch: bool = False
    ) -> Union[str, List[str], np.ndarray, List[np.ndarray]]:
        """
        Get all results from response.

        Args:
            stop_token: stop token for string generation
            is_batch: whether response is batched
        """
        process_result = lambda x: x.split(stop_token)[0] if stop_token else x
        extracted_items = [
            choice.text if isinstance(choice, LMModelChoice) else choice.array
            for choice in self._response.choices
        ]
        if len(extracted_items) == 0:
            return None
        if isinstance(extracted_items[0], str):
            processed_results = list(map(process_result, extracted_items))
        else:
            processed_results = extracted_items
        if len(processed_results) == 1 and not is_batch:
            return processed_results[0]
        else:
            return processed_results

    @classmethod
    def union_all(
        cls, responses: List["Response"], as_single_lmchoice: bool = False
    ) -> "Response":
        """Union a list of response.

        Args:
            responses: list of responses to union.
            as_single_lmchoice: if True, will concatenate all responses into a single
                model choice. Useful for merging streaming responses. Only valid
                for LMRequest responses.
        """
        if not responses:
            raise ValueError("Response list is empty.")
        if len(responses) == 1:
            return responses[0]
        first_response = responses[0]
        request_type = first_response._request_type
        response_type = first_response._response_type
        request = first_response.get_request_obj()

        if as_single_lmchoice and response_type != "text":
            raise ValueError("as_single_lmchoice=True only works for text responses.")

        # Make sure all responses have the same keys
        if not all(
            [
                (r._request_type == request_type)
                and (r._response_type == response_type)
                for r in responses
            ]
        ):
            raise ValueError("All responses must have the same keys.")

        # Get all the prompts and model choices
        all_prompts = []
        all_choices = []
        all_usages: List[Usage] = []
        all_engines = []
        for res in responses:
            all_engines.extend(res.get_request_obj().engine.split(ENGINE_SEP))
            res_prompt = res.get_request_obj().prompt
            if isinstance(res_prompt, str):
                res_prompt = [res_prompt]
            all_prompts.extend(res_prompt)
            all_choices.extend(res.get_response_obj().choices)
            if res.get_usage_obj().usages:
                all_usages.extend(res.get_usage_obj().usages)
            else:
                # Add empty usages if not present
                all_usages.extend([Usage()] * len(res_prompt))
        new_request = copy.deepcopy(request)
        new_request.engine = ENGINE_SEP.join(sorted(set(all_engines)))

        if as_single_lmchoice:
            if len(set(all_prompts)) != 1:
                raise ValueError("Prompts must be the same for as_single_lmchoice=True")
            all_choices_txt = cast(List[LMModelChoice], all_choices)  # type: ignore
            single_prompt = all_prompts[0]
            single_text = "".join([choice.text for choice in all_choices_txt])
            single_logprobs = [
                logprob
                for choice in all_choices_txt
                for logprob in choice.token_logprobs or []
            ]
            single_tokens = [
                token for choice in all_choices_txt for token in choice.tokens or []
            ]
            single_usage = Usage(
                completion_tokens=sum(usg.completion_tokens for usg in all_usages),
                prompt_tokens=sum(usg.prompt_tokens for usg in all_usages),
                total_tokens=sum(usg.total_tokens for usg in all_usages),
            )
            new_choices = [
                LMModelChoice(
                    text=single_text,
                    token_logprobs=single_logprobs,
                    tokens=single_tokens,
                )
            ]
            new_responses = ModelChoices(choices=new_choices)  # type: ignore
            new_usages = Usages(usages=[single_usage])
            new_request.prompt = single_prompt
            response_obj = cls(
                response=new_responses,
                cached=any(res.is_cached() for res in responses),
                request=new_request,
                usages=new_usages,
                request_type=request_type,
                response_type=response_type,
            )
            return response_obj
        else:
            new_request.prompt = all_prompts
            new_response = ModelChoices(choices=all_choices)
            new_usages = Usages(usages=all_usages)
            response_obj = cls(
                response=new_response,
                cached=any(res.is_cached() for res in responses),
                request=new_request,
                usages=new_usages,
                request_type=request_type,
                response_type=response_type,
            )
            return response_obj

    # Return a token by token iterator over the response
    def as_iter(self) -> Generator["Response", None, None]:
        """Return a token by token iterator over the response.

        Will return iterator of responses with one token each.
        """
        if self._response_type not in {"text"}:
            raise ValueError(
                f"Invalid response type {self._response_type} for as_iter()"
            )
        if not self._response.choices:
            raise ValueError("No choices in response.")
        if len(self._response.choices) > 1:
            raise ValueError(
                "Response has more than one choice. as_iter() "
                "should be over single choice responses."
            )
        if not isinstance(self._response.choices[0], LMModelChoice):
            raise ValueError(
                "response_type is text but response is "
                f"{self._response.choices[0].__class__}"
            )
        choice = cast(LMModelChoice, self._response.choices[0])
        # If tokens, return iterator of tokens
        if choice.tokens:
            for token, logprob in zip(choice.tokens, choice.token_logprobs):
                yield Response(
                    response=ModelChoices(
                        choices=[
                            LMModelChoice(
                                text=token, token_logprobs=[logprob], tokens=[token]
                            )
                        ]
                    ),
                    cached=self._cached,
                    request=self._request,
                    usages=self._usages,
                    request_type=self._request_type,
                    response_type=self._response_type,
                )
        # Otherwise, do it by words
        else:
            for i, word in enumerate(choice.text.split(" ")):
                word = " " + word if i > 0 else word
                yield Response(
                    response=ModelChoices(
                        choices=[
                            LMModelChoice(text=word, token_logprobs=None, tokens=None)
                        ]
                    ),
                    cached=self._cached,
                    request=self._request,
                    usages=self._usages,
                    request_type=self._request_type,
                    response_type=self._response_type,
                )

    def serialize(self) -> str:
        """
        Serialize response to string.

        Returns:
            serialized response.
        """
        return json.dumps(self.to_dict(), sort_keys=True, cls=NumpyArrayEncoder)

    @classmethod
    def deserialize(cls, value: str) -> "Response":
        """
        Deserialize string to response.

        Args:
            value: serialized response.

        Returns:
            serialized response.
        """
        deserialized = json.loads(value)
        return cls.from_dict(deserialized)

    def to_dict(self, drop_request: bool = False) -> Dict:
        """
        Get dictionary representation of response.

        Returns:
            dictionary representation of response.
        """
        to_return = {
            "response": self._response.dict(),
            "usages": self._usages.dict(),
            "cached": self._cached,
            "request": self._request.dict(),
            "response_type": self._response_type,
            "request_type": str(self._request_type.__name__),
            "item_dtype": self._item_dtype,
        }
        if drop_request:
            to_return.pop("request")
        return to_return

    @classmethod
    def from_dict(
        cls, response_dict: Dict, request_dict: Optional[Dict] = None
    ) -> "Response":
        """
        Create response from dictionary.

        Args:
            response: dictionary representation of response.
            request_dict: dictionary representation of request which
              will override what is in response_dict.

        Returns:
            response.
        """
        if "request" not in response_dict and request_dict is None:
            raise ValueError(
                "Request dictionary must be provided if "
                "request is not in response dictionary."
            )
        item_dtype = response_dict["item_dtype"]
        response_type = response_dict["response_type"]
        if response_dict["request_type"] == "LMRequest":
            request_type: Type[Request] = LMRequest
        elif response_dict["request_type"] == "LMChatRequest":
            request_type = LMChatRequest
        elif response_dict["request_type"] == "LMScoreRequest":
            request_type = LMScoreRequest
        elif response_dict["request_type"] == "EmbeddingRequest":
            request_type = EmbeddingRequest
        elif response_dict["request_type"] == "DiffusionRequest":
            request_type = DiffusionRequest
        choices: List[Union[LMModelChoice, ArrayModelChoice]] = []
        if item_dtype and response_type == "array":
            for choice in response_dict["response"]["choices"]:
                choice["array"] = np.array(choice["array"]).astype(item_dtype)
                choices.append(ArrayModelChoice(**choice))
        else:
            for choice in response_dict["response"]["choices"]:
                choices.append(LMModelChoice(**choice))
        response = ModelChoices(choices=choices)
        return cls(
            response=response,
            usages=Usages(**response_dict["usages"]),
            cached=response_dict["cached"],
            request=request_type(**(request_dict or response_dict["request"])),
            response_type=response_type,
            request_type=request_type,
        )

    def __str__(self) -> str:
        """
        Get string representation of response.

        Returns:
            string representation of response.
        """
        return self.serialize()

    def __repr__(self) -> str:
        """
        Get string representation of response.

        Returns:
            string representation of response.
        """
        return str(self)
