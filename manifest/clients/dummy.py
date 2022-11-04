"""Dummy client."""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from manifest.clients.client import Client

logger = logging.getLogger(__name__)

# User param -> (client param, default value)
DUMMY_PARAMS = {
    "n": ("num_results", 1),
}


class DummyClient(Client):
    """Dummy client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to dummpy server.

        This is a dummy client that returns identity responses. Used for testing.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        for key in DUMMY_PARAMS:
            setattr(self, key, client_args.pop(key, DUMMY_PARAMS[key][1]))

    def close(self) -> None:
        """Close the client."""
        pass

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        return {"engine": "dummy"}

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(DUMMY_PARAMS.keys())

    def get_request(
        self, query: str, request_args: Dict[str, Any] = {}
    ) -> Tuple[Callable[[], Dict], Dict]:
        """
        Get request string function.

        Args:
            query: query string.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        request_params = {"prompt": query}
        for key in DUMMY_PARAMS:
            request_params[DUMMY_PARAMS[key][0]] = request_args.pop(
                key, getattr(self, key)
            )

        def _run_completion() -> Dict:
            return {"choices": [{"text": "hello"}] * int(request_params["num_results"])}

        return _run_completion, request_params

    def get_choice_logit_request(
        self, query: str, gold_choices: List[str], request_args: Dict[str, Any] = {}
    ) -> Tuple[Callable[[], Dict], Dict]:
        """
        Get request string function for choosing max choices.

        Args:
            query: query string.
            gold_choices: choices for model to choose from via max logits.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        request_params = {"prompt": query, "gold_choices": gold_choices}

        def _run_completion() -> Dict:
            return {"choices": [{"text": gold_choices[0]}]}

        return _run_completion, request_params
