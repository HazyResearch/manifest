"""Client response."""
import json
from typing import Any, Dict, List, Union

import numpy as np

RESPONSE_CONSTRUCTORS = {
    "diffuser": {
        "logits_key": "token_logprobs",
        "item_key": "array",
    },
    "tomadiffuser": {
        "logits_key": "token_logprobs",
        "item_key": "array",
    },
    "openaiembedding": {
        "logits_key": "token_logprobs",
        "item_key": "array",
    },
}


class NumpyArrayEncoder(json.JSONEncoder):
    """Numpy array encoder."""

    def default(self, obj: Any) -> str:
        """Encode numpy array."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Response:
    """Response class."""

    def __init__(
        self,
        response: Dict,  # TODO: make pydantic model
        cached: bool,
        request_params: Dict,  # TODO: use request pydantic model
        generation_key: str = "choices",
        logits_key: str = "token_logprobs",
        item_key: str = "text",
        usage_key: str = "usage",
    ):
        """
        Initialize response.

        Args:
            response: response dict.
            cached: whether response is cached.
            request_params: request parameters.
            generation_key: key for generation results.
            logits_key: key for logits.
            item_key: key for item in the generations.
        """
        self.generation_key = generation_key
        self.logits_key = logits_key
        self.item_key = item_key
        self.usage_key = usage_key
        self.item_dtype = None
        if isinstance(response, dict):
            self._response = response
        else:
            raise ValueError(f"Response must be dict. Response is\n{response}.")
        if (
            (self.generation_key not in self._response)
            or (not isinstance(self._response[self.generation_key], list))
            or (len(self._response[self.generation_key]) <= 0)
        ):
            raise ValueError(
                "Response must be serialized to a dict with a nonempty"
                f" list of choices. Response is\n{self._response}."
            )
        # Turn off usage if it is not in response
        if self.usage_key not in self._response:
            self.usage_key = None
        else:
            if not isinstance(self._response[self.usage_key], list):
                raise ValueError(
                    "Response must be a list with usage dicts, one per choice."
                    f" Response is\n{self._response}."
                )

        if self.item_key not in self._response[self.generation_key][0]:
            raise ValueError(
                "Response must be serialized to a dict with a "
                f"list of choices with {self.item_key} field"
            )
        if (
            self.logits_key in self._response[self.generation_key][0]
            and self._response[self.generation_key][0][self.logits_key]
        ):
            if not isinstance(
                self._response[self.generation_key][0][self.logits_key], list
            ):
                raise ValueError(
                    f"{self.logits_key} must be a list of items "
                    "one for each token in the choice."
                )
        if isinstance(
            self._response[self.generation_key][0][self.item_key], np.ndarray
        ):
            self.item_dtype = str(
                self._response[self.generation_key][0][self.item_key].dtype
            )
        self._cached = cached
        self._request_params = request_params

    def is_cached(self) -> bool:
        """Check if response is cached."""
        return self._cached

    def get_request(self) -> Dict:
        """Get request parameters."""
        return self._request_params

    def get_json_response(self) -> Dict:
        """Get response dict without parsing."""
        return self._response

    def get_response(
        self, stop_token: str = "", is_batch: bool = False
    ) -> Union[str, List[str], np.ndarray, List[np.ndarray]]:
        """
        Get all results from response.

        Args:
            stop_token: stop token for string generation
            is_batch: whether response is batched
        """
        process_result = (
            lambda x: x.strip().split(stop_token)[0] if stop_token else x.strip()
        )
        extracted_items = [
            choice[self.item_key] for choice in self._response[self.generation_key]
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
        item_dtype = deserialized["item_dtype"]
        if item_dtype:
            for choice in deserialized["response"][deserialized["generation_key"]]:
                choice[deserialized["item_key"]] = np.array(
                    choice[deserialized["item_key"]]
                ).astype(item_dtype)
        return cls(
            deserialized["response"],
            deserialized["cached"],
            deserialized["request_params"],
            generation_key=deserialized["generation_key"],
            logits_key=deserialized["logits_key"],
            item_key=deserialized["item_key"],
        )

    def to_dict(self) -> Dict:
        """
        Get dictionary representation of response.

        Returns:
            dictionary representation of response.
        """
        return {
            "generation_key": self.generation_key,
            "logits_key": self.logits_key,
            "item_key": self.item_key,
            "item_dtype": self.item_dtype,
            "response": self._response,
            "cached": self._cached,
            "request_params": self._request_params,
        }

    @classmethod
    def from_dict(cls, response: Dict) -> "Response":
        """
        Create response from dictionary.

        Args:
            response: dictionary representation of response.

        Returns:
            response.
        """
        return cls(
            response["response"],
            response["cached"],
            response["request_params"],
            generation_key=response["generation_key"],
            logits_key=response["logits_key"],
            item_key=response["item_key"],
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
