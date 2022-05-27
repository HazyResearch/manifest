"""Model class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Model(ABC):
    """Model class."""

    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize model.

        kwargs are passed to model as default parameters.

        Args:
            model_name: model name string.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_init_params(self) -> Dict:
        """Return init params to determine what model is being used."""
        raise NotImplementedError()

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> List[str]:
        """
        Generate the prompt from model.

        Outputs must be generated text, not including prompt.

        Args:
            prompt: promt to generate from.

        Returns:
            list of generated text (list of length 1 for 1 generation).
        """
        raise NotImplementedError()
