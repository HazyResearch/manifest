"""Client class."""
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

from pyChatGPT import ChatGPT

from manifest.clients.client import Client
from manifest.request import LMRequest, Request

logger = logging.getLogger(__name__)


class ChatGPTClient(Client):
    """ChatGPT Client class."""

    # No params for ChatGPT
    PARAMS: Dict[str, Tuple[str, Any]] = {}
    REQUEST_CLS = LMRequest

    def connect(
        self, connection_str: Optional[str], client_args: Dict[str, Any]
    ) -> None:
        """
        Connect to ChatGPT.

        We use https://github.com/terry3041/pyChatGPT.

        Arsg:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.session_key = os.environ.get("CHATGPT_SESSION_KEY", connection_str)
        if self.session_key is None:
            raise ValueError(
                "ChatGPT session key not set. Set CHATGPT_SESSION_KEY environment "
                "variable or pass through `client_connection`. "
                "For details, see https://github.com/terry3041/pyChatGPT "
                "and go through instructions for getting a session key."
            )
        self.host = None
        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))
        self._chat_session = ChatGPT(self.session_key, verbose=False)

    def close(self) -> None:
        """Close the client."""
        self._chat_session = None

    def clear_conversations(self) -> None:
        """Clear conversations.

        Only works for ChatGPT.
        """
        self._chat_session.clear_conversations()

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return ""

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return {}

    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference."""
        return False

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        return {"model_name": "chatgpt", "engine": "chatgpt"}

    def format_response(self, response: Dict) -> Dict[str, Any]:
        """
        Format response to dict.

        Args:
            response: response

        Return:
            response as dict
        """
        return {
            "model": "chatgpt",
            "choices": [
                {
                    "text": response["message"],
                }
            ],
        }

    def get_request(self, request: Request) -> Tuple[Callable[[], Dict], Dict]:
        """
        Get request string function.

        Args:
            request: request.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        if isinstance(request.prompt, list):
            raise ValueError("ChatGPT does not support batch inference.")

        prompt = str(request.prompt)
        request_params = request.to_dict(self.PARAMS)

        def _run_completion() -> Dict:
            try:
                res = self._chat_session.send_message(prompt)
            except Exception as e:
                logger.error(f"ChatGPT error {e}.")
                raise e
            return self.format_response(res)

        return _run_completion, request_params
