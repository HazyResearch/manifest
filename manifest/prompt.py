"""Prompt class."""

import inspect
import logging
from typing import Any, Callable, List, Optional, Union

import dill

logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class Prompt:
    """Prompt class."""

    def __init__(self, prompt_obj: Union[str, Callable, "Prompt", List["Prompt"]]):
        """
        Initialize prompt.

        If prompt_obj is a string, it will be cast as function.
        If prompt_obj is list of promts, it will be composed.
        """
        # TODO: figure out how to compose prompts to keep the
        # interface simple? Can we make a function
        # such that a single call will run the composition?
        if isinstance(prompt_obj, str):
            self.prompt_func = lambda: prompt_obj
        elif callable(prompt_obj):
            self.prompt_func = prompt_obj
        else:
            # TODO: implement
            raise NotImplementedError()
        self.num_args = len(inspect.signature(self.prompt_func).parameters)
        if self.num_args > 1:
            raise ValueError("Prompts must have zero or one input.")

    def __call__(self, input: Optional[Any] = None) -> str:
        """
        Return the prompt given the inputs.

        Args:
            input: input to prompt.

        Returns:
            prompt string.
        """
        if self.num_args >= 1:
            return self.prompt_func(input)  # type: ignore
        else:
            return self.prompt_func()

    def serialize(self) -> str:
        """
        Return the prompt as str.

        Returns:
            prompt as str.
        """
        return dill.dumps(self.prompt_func).decode("latin1")

    @classmethod
    def deserialize(cls, obj: str) -> "Prompt":
        """
        Return the prompt from a json object.

        Args:
            obj: json object.

        Return:
            prompt.
        """
        return Prompt(dill.loads(obj.encode("latin1")))
