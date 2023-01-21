"""Client class."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests

from manifest.request import Request

logger = logging.getLogger(__name__)


class Client(ABC):
    """Client class."""

    # Must be overridden by child class
    PARAMS: Dict[str, Tuple[str, Any]] = {}
    REQUEST_CLS = Request

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

    def get_request_params(
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
        return self.REQUEST_CLS(**params)

    def format_response(self, response: Dict) -> Dict[str, Any]:
        """
        Format response to dict.

        Args:
            response: response

        Return:
            response as dict
        """
        if "choices" not in response:
            raise ValueError(f"Invalid response: {response}")
        return response

    def get_request(self, request: Request) -> Tuple[Callable[[], Dict], Dict]:
        """
        Get request string function.

        Args:
            request: request.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        if isinstance(request.prompt, list) and not self.supports_batch_inference():
            raise ValueError(
                f"{self.__class__.__name__} does not support batch inference."
            )

        request_params = request.to_dict(self.PARAMS)
        retry_timeout = request_params.pop("client_timeout")

        def _run_completion() -> Dict:
            post_str = self.get_generation_url()
            try:
                res = requests.post(
                    post_str,
                    headers=self.get_generation_header(),
                    json=request_params,
                    timeout=retry_timeout,
                )
                res.raise_for_status()
            except requests.Timeout as e:
                logger.error(
                    f"{self.__class__.__name__} request timed out."
                    " Increase client_timeout."
                )
                raise e
            except requests.exceptions.HTTPError:
                logger.error(res.json())
                raise requests.exceptions.HTTPError(res.json())
            return self.format_response(res.json())

        return _run_completion, request_params

    def get_score_prompt_request(
        self,
        request: Request,
    ) -> Tuple[Callable[[], Dict], Dict]:
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
