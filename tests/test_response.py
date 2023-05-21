"""Response test."""
from typing import List, cast

import numpy as np
import pytest

from manifest import Response
from manifest.request import EmbeddingRequest, LMRequest
from manifest.response import (
    ArrayModelChoice,
    LMModelChoice,
    ModelChoices,
    Usage,
    Usages,
)


def test_init(
    model_choice: ModelChoices,
    model_choice_arr: ModelChoices,
    model_choice_arr_int: ModelChoices,
    request_lm: LMRequest,
    request_array: EmbeddingRequest,
) -> None:
    """Test response initialization."""
    response = Response(
        response=model_choice,
        cached=False,
        request=request_lm,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )
    assert response._response == model_choice
    assert response._cached is False
    assert response._request == request_lm
    assert response._usages == Usages(usages=[])
    assert response._request_type == LMRequest
    assert response._response_type == "text"
    assert response._item_dtype is None

    response = Response(
        response=model_choice_arr_int,
        cached=False,
        request=request_array,
        usages=Usages(usages=[Usage(total_tokens=4), Usage(total_tokens=6)]),
        request_type=EmbeddingRequest,
        response_type="array",
    )
    assert response._cached is False
    assert response._request == request_array
    assert sum([usg.total_tokens for usg in response._usages.usages]) == 10
    assert response._request_type == EmbeddingRequest
    assert response._response_type == "array"
    assert response._item_dtype == "int64"

    with pytest.raises(ValueError) as excinfo:
        Response(
            response=model_choice,
            cached=False,
            request=request_lm,
            usages=None,
            request_type=LMRequest,
            response_type="blah",
        )
    assert "blah" in str(excinfo.value)

    # Can't convert array with text
    with pytest.raises(ValueError) as excinfo:
        Response(
            response=model_choice,
            cached=False,
            request=request_lm,
            usages=None,
            request_type=LMRequest,
            response_type="array",
        )
    assert str(excinfo.value) == (
        "response_type is array but response is "
        "<class 'manifest.response.LMModelChoice'>"
    )

    # Can't convert text with array
    with pytest.raises(ValueError) as excinfo:
        Response(
            response=model_choice_arr,
            cached=False,
            request=request_array,
            usages=None,
            request_type=LMRequest,
            response_type="text",
        )
    assert str(excinfo.value) == (
        "response_type is text but response is "
        "<class 'manifest.response.ArrayModelChoice'>"
    )


def test_getters(model_choice: ModelChoices, request_lm: LMRequest) -> None:
    """Test response cached."""
    response = Response(
        response=model_choice,
        cached=False,
        request=request_lm,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )
    assert response.get_response_obj() == model_choice
    assert response.is_cached() is False
    assert response.get_request_obj() == request_lm
    assert response.get_usage_obj() == Usages(usages=[])
    assert response.get_json_response() == model_choice.dict()
    assert response.get_response() == ["hello", "bye"]


def test_serialize(
    model_choice: ModelChoices,
    model_choice_arr: ModelChoices,
    model_choice_arr_int: ModelChoices,
    request_lm: LMRequest,
    request_array: EmbeddingRequest,
) -> None:
    """Test response serialization."""
    response = Response(
        response=model_choice,
        cached=False,
        request=request_lm,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )
    deserialized_response = Response.deserialize(response.serialize())
    assert deserialized_response.get_response_obj() == model_choice
    assert deserialized_response.is_cached() is False
    assert deserialized_response.get_request_obj() == request_lm
    assert deserialized_response.get_usage_obj() == Usages(usages=[])
    assert deserialized_response.get_json_response() == model_choice.dict()
    assert deserialized_response.get_response() == ["hello", "bye"]

    deserialized_response = Response.from_dict(response.to_dict())
    assert deserialized_response.get_response_obj() == model_choice
    assert deserialized_response.is_cached() is False
    assert deserialized_response.get_request_obj() == request_lm
    assert deserialized_response.get_usage_obj() == Usages(usages=[])
    assert deserialized_response.get_json_response() == model_choice.dict()
    assert deserialized_response.get_response() == ["hello", "bye"]

    deserialized_response = Response.from_dict(
        response.to_dict(drop_request=True), request_dict={"prompt": "blahhhh"}
    )
    assert deserialized_response.get_response_obj() == model_choice
    assert deserialized_response.is_cached() is False
    assert deserialized_response.get_request_obj().prompt == "blahhhh"
    assert deserialized_response.get_usage_obj() == Usages(usages=[])
    assert deserialized_response.get_json_response() == model_choice.dict()
    assert deserialized_response.get_response() == ["hello", "bye"]

    # Int type
    response = Response(
        response=model_choice_arr_int,
        cached=False,
        request=request_array,
        usages=Usages(usages=[Usage(total_tokens=4), Usage(total_tokens=6)]),
        request_type=EmbeddingRequest,
        response_type="array",
    )
    deserialized_response = Response.deserialize(response.serialize())
    assert deserialized_response._item_dtype == "int64"
    assert (
        cast(
            ArrayModelChoice, deserialized_response.get_response_obj().choices[0]
        ).array.dtype
        == np.int64
    )
    assert np.array_equal(
        cast(
            ArrayModelChoice, deserialized_response.get_response_obj().choices[0]
        ).array,
        cast(ArrayModelChoice, model_choice_arr_int.choices[0]).array,
    )

    # Float type
    response = Response(
        response=model_choice_arr,
        cached=False,
        request=request_array,
        usages=Usages(usages=[Usage(total_tokens=4), Usage(total_tokens=6)]),
        request_type=EmbeddingRequest,
        response_type="array",
    )
    deserialized_response = Response.deserialize(response.serialize())
    assert deserialized_response._item_dtype == "float64"
    assert (
        cast(
            ArrayModelChoice, deserialized_response.get_response_obj().choices[0]
        ).array.dtype
        == np.float64
    )
    assert np.array_equal(
        cast(
            ArrayModelChoice, deserialized_response.get_response_obj().choices[0]
        ).array,
        cast(ArrayModelChoice, model_choice_arr.choices[0]).array,
    )


def test_get_results(
    model_choice: ModelChoices,
    model_choice_single: ModelChoices,
    model_choice_arr: ModelChoices,
    request_lm: LMRequest,
    request_array: EmbeddingRequest,
) -> None:
    """Test response get results."""
    response = Response(
        response=model_choice_single,
        cached=False,
        request=request_lm,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )
    assert response.get_response() == "helloo"
    assert response.get_response(stop_token="ll") == "he"
    assert response.get_response(stop_token="ll", is_batch=True) == ["he"]

    response = Response(
        response=model_choice,
        cached=False,
        request=request_lm,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )
    assert response.get_response() == ["hello", "bye"]
    assert response.get_response(stop_token="b") == ["hello", ""]
    assert response.get_response(stop_token="y", is_batch=True) == ["hello", "b"]

    float_arr1 = cast(ArrayModelChoice, model_choice_arr.choices[0]).array
    float_arr2 = cast(ArrayModelChoice, model_choice_arr.choices[1]).array
    response = Response(
        response=model_choice_arr,
        cached=False,
        request=request_array,
        usages=Usages(usages=[Usage(total_tokens=4), Usage(total_tokens=6)]),
        request_type=EmbeddingRequest,
        response_type="array",
    )
    assert np.array_equal(response.get_response()[0], float_arr1)
    assert np.array_equal(response.get_response()[1], float_arr2)
    assert np.array_equal(response.get_response(stop_token="t")[0], float_arr1)
    assert np.array_equal(response.get_response(stop_token="t")[1], float_arr2)


def test_union_all(
    model_choice: ModelChoices,
    model_choice_single: ModelChoices,
    request_lm: LMRequest,
    request_lm_single: LMRequest,
) -> None:
    """Test union all."""
    response1 = Response(
        response=model_choice,
        cached=False,
        request=request_lm,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )

    response2 = Response(
        response=model_choice_single,
        cached=False,
        request=request_lm_single,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )

    final_response = Response.union_all([response1, response2])
    assert final_response.get_json_response() == {
        "choices": [
            {"text": "hello", "token_logprobs": [0.1, 0.2], "tokens": ["hel", "lo"]},
            {"text": "bye", "token_logprobs": [0.3], "tokens": ["bye"]},
            {"text": "helloo", "token_logprobs": [0.1, 0.2], "tokens": ["hel", "loo"]},
        ]
    }
    assert final_response.get_usage_obj() == Usages(usages=[Usage(), Usage(), Usage()])
    merged_prompts: List[str] = request_lm.prompt + [request_lm_single.prompt]  # type: ignore  # noqa: E501
    assert final_response.get_request_obj().prompt == merged_prompts
    assert final_response.get_request_obj().engine == "dummy::text-ada-001"

    # Modify A to have usage and cached
    response1 = Response(
        response=model_choice,
        cached=False,
        request=request_lm,
        usages=Usages(usages=[Usage(total_tokens=4), Usage(total_tokens=6)]),
        request_type=LMRequest,
        response_type="text",
    )

    final_response = Response.union_all([response1, response2])
    assert final_response.get_usage_obj() == Usages(
        usages=[Usage(total_tokens=4), Usage(total_tokens=6), Usage()]
    )

    # Test merge to single
    model_choices = ModelChoices(
        choices=[
            LMModelChoice(
                text=" helloo this is a bug",
                token_logprobs=[0.1, 0.2, 0.3],
                tokens=[" helloo", " this is", " a bug"],
            ),
        ]
    )
    request = LMRequest(prompt="monkey", engine="dummy")
    response1 = Response(
        response=model_choices,
        cached=False,
        request=request,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )
    final_response = Response.union_all([response1, response1], as_single_lmchoice=True)
    assert final_response.get_json_response() == {
        "choices": [
            {
                "text": " helloo this is a bug helloo this is a bug",
                "token_logprobs": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
                "tokens": [
                    " helloo",
                    " this is",
                    " a bug",
                    " helloo",
                    " this is",
                    " a bug",
                ],
            },
        ]
    }
    assert final_response.get_usage_obj() == Usages(usages=[Usage()])
    assert final_response.get_request_obj().prompt == "monkey"
    assert final_response.get_request_obj().engine == "dummy"


def test_as_iter(
    model_choice_single: ModelChoices, request_lm_single: LMRequest
) -> None:
    """Test as iter."""
    response = Response(
        response=model_choice_single,
        cached=False,
        request=request_lm_single,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )
    response_iter_list = list(response.as_iter())
    assert len(response_iter_list) == 2
    assert response_iter_list[0].get_response() == "hel"
    assert response_iter_list[1].get_response() == "loo"

    model_choices = ModelChoices(
        choices=[
            LMModelChoice(text="helloo this is a bug"),
        ]
    )
    request = LMRequest(prompt="monkey", engine="dummy")
    response = Response(
        response=model_choices,
        cached=False,
        request=request,
        usages=None,
        request_type=LMRequest,
        response_type="text",
    )
    response_iter_list = list(response.as_iter())
    assert len(response_iter_list) == 5
    assert response_iter_list[0].get_response() == "helloo"
    assert response_iter_list[1].get_response() == " this"
    assert response_iter_list[2].get_response() == " is"
    assert response_iter_list[3].get_response() == " a"
    assert response_iter_list[4].get_response() == " bug"
