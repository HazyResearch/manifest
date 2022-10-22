"""Model class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class Model(ABC):
    """Model class."""

    @abstractmethod
    def __init__(
        self,
        model_name_or_path: str,
        model_config: str,
        cache_dir: str,
        device: int,
        use_accelerate: bool,
        use_parallelize: bool,
        use_bitsandbytes: bool,
        perc_max_gpu_mem_red: float,
        use_fp16: bool,
    ):
        """
        Initialize model.

        All arguments will be passed in the request from Manifest.

        Args:
            model_name_or_path: model name string.
            model_config: model config string.
            cache_dir: cache directory for model.
            device: device to use for model.
            use_accelerate: whether to use accelerate for multi-gpu inference.
            use_parallelize: use HF default parallelize
            use_bitsandbytes: use HF bits and bytes
            perc_max_gpu_mem_red: percent max memory reduction in accelerate
            use_fp16: use fp16 for model weights.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_init_params(self) -> Dict:
        """Return init params to determine what model is being used."""
        raise NotImplementedError()

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> List[Tuple[str, float]]:
        """
        Generate the prompt from model.

        Outputs must be generated text and score, not including prompt.

        Args:
            prompt: promt to generate from.

        Returns:
            list of generated text (list of length 1 for 1 generation).
        """
        raise NotImplementedError()

    @abstractmethod
    def logits_scoring(
        self, prompt: str, gold_choices: List[str], **kwargs: Any
    ) -> Tuple[str, float]:
        """
        Given the prompt and gold choices, choose the best choice with max logits.

        Args:
            prompt: promt to generate from.
            gold_choices: list of choices to choose from.

        Returns:
            the returned gold choice and the score.
        """
        raise NotImplementedError()
