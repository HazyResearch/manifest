"""Client response."""
import json
from typing import Dict, List, Union


class Response:
    """Response class."""

    def __init__(self, response: Union[str, Dict]):
        """Initialize response."""
        if isinstance(response, str):
            self.response = json.loads(response)
        elif isinstance(response, dict):
            self.response = response
        else:
            raise ValueError("Response must be str or dict")
        if ("choices" not in self.response) or (
            not isinstance(self.response["choices"], list)
        ):
            raise ValueError(
                "Response must be serialized to a dict with a list of choices"
            )
        if len(self.response["choices"]) > 0:
            if "text" not in self.response["choices"][0]:
                raise ValueError(
                    "Response must be serialized to a dict with a "
                    "list of choices with text field"
                )

    def __getitem__(self, key: str) -> str:
        """
        Return the response given the key.

        Args:
            key: key to get.

        Returns:
            value of key.
        """
        return self.response[key]

    def get_results(self) -> Union[str, List[str]]:
        """Get all text results from response."""
        if len(self.response["choices"]) == 0:
            return None
        if len(self.response["choices"]) == 1:
            return self.response["choices"][0]["text"]
        return [choice["text"] for choice in self.response["choices"]]

    def serialize(self) -> str:
        """
        Serialize response to string.

        Returns:
            serialized response.
        """
        return json.dumps(self.response, sort_keys=True)

    @classmethod
    def deserialize(cls, value: str) -> "Response":
        """
        Deserialize string to response.

        Args:
            value: serialized response.

        Returns:
            serialized response.
        """
        return Response(value)
