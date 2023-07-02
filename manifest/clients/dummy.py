"""Dummy client."""
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tiktoken

from manifest.clients.client import Client
from manifest.request import LMChatRequest, LMRequest, LMScoreRequest, Request
from manifest.response import LMModelChoice, ModelChoices, Response, Usage, Usages

logger = logging.getLogger(__name__)


class DummyClient(Client):
    """Dummy client."""

    # User param -> (client param, default value)
    PARAMS = {
        "engine": ("model", "text-davinci-003"),
        "temperature": ("temperature", 0.0),
        "max_tokens": ("max_tokens", 10),
        "n": ("n", 1),
        "top_p": ("top_p", 1.0),
        "top_k": ("best_of", 1),
        "batch_size": ("batch_size", 20),
    }
    REQUEST_CLS = LMRequest
    NAME = "dummy"

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
        # We tiktoken as it is faster than HF for tokenizing
        # Use any model to create the tokenizer
        self.encoder = tiktoken.get_encoding("cl100k_base")
        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))

    def close(self) -> None:
        """Close the client."""
        pass

    def get_generation_url(self) -> str:
        """Get generation URL."""
        return "dummy"

    def supports_batch_inference(self) -> bool:
        """Return whether the client supports batch inference."""
        return True

    def supports_streaming_inference(self) -> bool:
        """Return whether the client supports streaming inference.

        Override in child client class.
        """
        return False

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return {}

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        return {"engine": "dummy", "model": getattr(self, "engine")}

    def get_mock_output(
        self, output_toks: int, is_completion: bool, seed: Optional[int] = None
    ) -> LMModelChoice:
        """Return mock model output by generating random tokens."""
        np.random.seed(seed)
        random_tokens = np.random.randint(
            0, self.encoder.max_token_value + 1, output_toks
        )
        response = self.encoder.decode(random_tokens)  # type: ignore
        if is_completion:
            np.random.seed(seed)
            random_logprobs = np.random.uniform(
                low=-2, high=-0.00001, size=output_toks
            ).tolist()
        else:
            # Return all Nones to mimic chat models
            # OpenAI chat models do not return logprobs
            random_logprobs = [None] * output_toks
        return LMModelChoice(
            text=response,
            token_logprobs=random_logprobs,
            tokens=random_tokens.tolist(),
        )

    def get_mock_choices(
        self,
        prompt_list: List[str],
        request_params: Dict,
        is_completion: bool,
    ) -> Tuple[List[LMModelChoice], List[Usage]]:
        """Get choices and usages of mock output."""
        choices = []
        usages = []
        for prompt in prompt_list:
            num_prompt_tokens = len(self.encoder.encode(prompt))
            if request_params["temperature"] == 0:
                # Get integer seed from hash of prompt
                seed = (
                    int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
                    % 10**8
                )
            else:
                # Get random seed
                seed = None
            for _ in range(int(request_params["n"])):
                choice = self.get_mock_output(
                    request_params["max_tokens"], is_completion=is_completion, seed=seed
                )
                choices.append(choice)
                usages.append(
                    Usage(
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=request_params["max_tokens"],
                        total_tokens=num_prompt_tokens + request_params["max_tokens"],
                    )
                )
        return choices, usages

    def run_request(self, request: Request) -> Response:
        """
        Get request string function.

        Args:
            request: request.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        if isinstance(request.prompt, list):
            prompt_list = request.prompt
        else:
            prompt_list = [request.prompt]
        request_params = request.to_dict(self.PARAMS)

        choices, usages = self.get_mock_choices(
            prompt_list, request_params, is_completion=True
        )
        return Response(
            response=ModelChoices(choices=choices),  # type: ignore
            cached=False,
            request=request,
            usages=Usages(usages=usages),
            response_type="text",
            request_type=self.REQUEST_CLS,
        )

    async def arun_batch_request(
        self, request: Request, verbose: bool = False
    ) -> Response:
        """
        Get async request string function.

        Args:
            request: request.

        Returns:
            response.
        """
        return self.run_request(request)

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
        prompt_list = ["_".join(pmp["content"] for pmp in request.prompt)]
        request_params = request.to_dict(self.PARAMS)

        choices, usages = self.get_mock_choices(
            prompt_list, request_params, is_completion=False
        )
        return Response(
            response=ModelChoices(choices=choices),  # type: ignore
            cached=False,
            request=request,
            usages=Usages(usages=usages),
            response_type="text",
            request_type=LMChatRequest,
        )

    def run_score_prompt_request(
        self,
        request: LMScoreRequest,
    ) -> Response:
        """
        Get the logit score of the prompt via a forward pass of the model.

        Args:
            request: request.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        if isinstance(request.prompt, list):
            prompt_list = request.prompt
        else:
            prompt_list = [request.prompt]
        request_params = request.to_dict(self.PARAMS)

        choices, usages = self.get_mock_choices(
            prompt_list, request_params, is_completion=True
        )
        return Response(
            response=ModelChoices(choices=choices),  # type: ignore
            cached=False,
            request=request,
            usages=Usages(usages=usages),
            response_type="text",
            request_type=LMScoreRequest,
        )
