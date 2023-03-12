"""Model class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np


class Model(ABC):
    """Model class."""

    @abstractmethod
    def __init__(
        self,
        model_name_or_path: str,
        model_type: str,
        cache_dir: str,
        device: int,
        use_accelerate: bool,
        use_parallelize: bool,
        use_bitsandbytes: bool,
        use_deepspeed: bool,
        perc_max_gpu_mem_red: float,
        use_fp16: bool,
    ):
        """
        Initialize model.

        All arguments will be passed in the request from Manifest.

        Args:
            model_name_or_path: model name string.
            model_type: model type string for when model_name not in registry.
            cache_dir: cache directory for model.
            device: device to use for model.
            use_accelerate: whether to use accelerate for multi-gpu inference.
            use_parallelize: use HF default parallelize
            use_bitsandbytes: use HF bits and bytes
            use_deepspeed: use deepspeed
            perc_max_gpu_mem_red: percent max memory reduction in accelerate
            use_fp16: use fp16 for model weights.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_init_params(self) -> Dict:
        """Return init params to determine what model is being used."""
        raise NotImplementedError()

    def generate(
        self, prompt: Union[str, List[str]], **kwargs: Any
    ) -> List[Tuple[Any, float, List[int], List[float]]]:
        """
        Generate the prompt from model.

        Outputs must be generated text and score, not including prompt.

        Args:
            prompt: promt to generate from.

        Returns:
            list of generated text (list of length 1 for 1 generation).
            Each item is the response, answer logprob, list of tokens,
            and list of logprobs for each token.
        """
        raise NotImplementedError()

    def embed(self, prompt: Union[str, List[str]], **kwargs: Any) -> np.ndarray:
        """
        Compute embedding for prompts.

        Args:
            prompt: promt to generate from.

        Returns:
            embedding
        """
        raise NotImplementedError()

    def score_sequence(
        self, prompt: Union[str, List[str]], **kwargs: Any
    ) -> List[Tuple[float, List[int], List[float]]]:
        """
        Score a sequence of choices.

        Args:
            prompt (:obj:`str` or :obj:`List[str]`):
                The prompt to score the choices against.
            **kwargs:
                Additional keyword arguments passed along to the :obj:`__call__` method.

        Returns:
            Tuple of total score, tokens, and probs per token.
        """
        raise NotImplementedError()
