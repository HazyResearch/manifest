"""Zoo model."""
import os
import sys
from typing import Any, Dict, List, Tuple

from manifest.api.models.model import Model

ZOO_PATH = os.environ.get("ZOO_PATH", None)
if not ZOO_PATH:
    raise ImportError("ZOO_PATH environment variable not set.")
sys.path.append(ZOO_PATH)

from src.models.s4_seq import S4LMManifest  # type: ignore


class ZooModel(Model):
    """Zoo model."""

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
            model_config: model config path.
            cache_dir: cache directory for model.
            device: device to use for model.
            use_accelerate: whether to use accelerate for multi-gpu inference.
            use_parallelize: use HF default parallelize
            use_bitsandbytes: use HF bits and bytes
            perc_max_gpu_mem_red: percent max memory reduction in accelerate
            use_fp16: use fp16 for model weights.
        """
        # Check if providing path
        self.model_path = model_name_or_path
        self.model_config = model_config
        if not self.model_config:
            raise ValueError("Must provide model config.")
        self.model = S4LMManifest(
            config_path=self.model_config,
            weights_path=self.model_path,
        )
        # Can only load this after the model has been initialized
        self.model_name = self.model.get_model_name()

    def get_init_params(self) -> Dict:
        """Return init params to determine what model is being used."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_config": self.model_config,
        }

    def generate(self, prompt: str, **kwargs: Any) -> List[Tuple[str, float]]:
        """
        Generate the prompt from model.

        Outputs must be generated text and score, not including prompt.

        Args:
            prompt: promt to generate from.

        Returns:
            list of generated text (list of length 1 for 1 generation).
        """
        print(prompt)
        final_results = self.model.generate(prompt, **kwargs)
        return final_results

    def logits_scoring(
        self, prompt: str, gold_choices: List[str], **kwargs: Any
    ) -> Tuple[str, float]:
        """
        Given the prompt and gold choices, choose the best choice with max logits.

        Args:
            prompt: promt to generate from.
            gold_choices: list of choices to choose from.

        Returns:
            the returned gold choice and the score
        """
        raise NotImplementedError()
