"""Client response."""
import json
from typing import Dict, List, Union


class Response:
    """Response class."""

    def __init__(self, response: Dict, cached: bool, request_params: Dict):
        """Initialize response."""
        if isinstance(response, dict):
            self._response = response
        else:
            raise ValueError(f"Response must be str or dict. Response is\n{response}.")
        if ("choices" not in self._response) or (
            not isinstance(self._response["choices"], list)
        ):
            raise ValueError(
                "Response must be serialized to a dict with a list of choices. "
                f"Response is\n{self._response}."
            )
        if len(self._response["choices"]) > 0:
            if "text" not in self._response["choices"][0]:
                raise ValueError(
                    "Response must be serialized to a dict with a "
                    "list of choices with text field"
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

    def get_response(self, stop_token: str = "") -> Union[str, List[str], None]:
        """
        Get all text results from response.

        Args:
            stop_token: stop token for string generation
        """
        process_result = (
            lambda x: x.strip().split(stop_token)[0] if stop_token else x.strip()
        )
        if len(self._response["choices"]) == 0:
            return None
        if len(self._response["choices"]) == 1:
            return process_result(self._response["choices"][0]["text"])
        return [process_result(choice["text"]) for choice in self._response["choices"]]

    def serialize(self) -> str:
        """
        Serialize response to string.

        Returns:
            serialized response.
        """
        return json.dumps(self.to_dict(), sort_keys=True)

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
        return cls(
            deserialized["response"],
            deserialized["cached"],
            deserialized["request_params"],
        )

    def to_dict(self) -> Dict:
        """
        Get dictionary representation of response.

        Returns:
            dictionary representation of response.
        """
        return {
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
