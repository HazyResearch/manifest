"""Sentence transformer model."""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from manifest.api.models.model import Model


class SentenceTransformerModel(Model):
    """SentenceTransformer model."""

    def __init__(
        self,
        model_name_or_path: str,
        model_type: Optional[str] = None,
        model_config: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: int = 0,
        use_accelerate: bool = False,
        use_parallelize: bool = False,
        use_bitsandbytes: bool = False,
        use_deepspeed: bool = False,
        perc_max_gpu_mem_red: float = 1.0,
        use_fp16: bool = False,
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
            use_deepspeed: use deepspeed
            perc_max_gpu_mem_red: percent max memory reduction in accelerate
            use_fp16: use fp16 for model weights.
        """
        if use_accelerate or use_parallelize or use_bitsandbytes or use_deepspeed:
            raise ValueError(
                "Cannot use accelerate or parallelize or "
                "bitsandbytes or deepspeeed with sentence transformers"
            )
        # Check if providing path
        self.model_name = model_name_or_path
        print("Model Name:", self.model_name)
        torch_device = (
            torch.device("cpu")
            if (device == -1 or not torch.cuda.is_available())
            else torch.device(f"cuda:{device}")
        )
        self.embedding_model = SentenceTransformer(self.model_name, device=torch_device)
        self.embedding_model.to(torch_device)
        self.embedding_model.eval()

    def get_init_params(self) -> Dict:
        """Return init params to determine what model is being used."""
        return {"model_name": self.model_name, "model_path": self.model_name}

    @torch.no_grad()
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
        """
        raise NotImplementedError("Generate not supported for sentence transformers")

    @torch.no_grad()
    def embed(self, prompt: Union[str, List[str]], **kwargs: Any) -> np.ndarray:
        """
        Embed the prompt from model.

        Args:
            prompt: promt to embed from.

        Returns:
            list of embeddings (list of length 1 for 1 embedding).
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        return self.embedding_model.encode(prompt)

    @torch.no_grad()
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
        """
        raise NotImplementedError(
            "Score sequence not supported for sentence transformers"
        )
