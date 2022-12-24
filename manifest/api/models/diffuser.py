"""Huggingface model."""
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from diffusers import StableDiffusionPipeline

from manifest.api.models.model import Model


class DiffuserModel(Model):
    """Diffuser model."""

    def __init__(
        self,
        model_name_or_path: str,
        model_config: str = None,
        cache_dir: str = None,
        device: int = 0,
        use_accelerate: bool = False,
        use_parallelize: bool = False,
        use_bitsandbytes: bool = False,
        perc_max_gpu_mem_red: float = 1.0,
        use_fp16: bool = True,
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
        if use_accelerate or use_parallelize or use_bitsandbytes:
            raise ValueError(
                "Cannot use accelerate or parallelize or bitsandbytes with diffusers"
            )
        # Check if providing path
        self.model_path = model_name_or_path
        if Path(self.model_path).exists() and Path(self.model_path).is_dir():
            model_name_or_path = Path(self.model_path).name
        self.model_name = model_name_or_path
        print("Model Name:", self.model_name, "Model Path:", self.model_path)
        dtype = torch.float16 if use_fp16 else None
        torch_device = (
            torch.device("cpu")
            if (device == -1 or not torch.cuda.is_available())
            else torch.device(f"cuda:{device}")
        )
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            revision="fp16" if str(dtype) == "float16" else None,
        )
        self.pipeline.safety_checker = None
        self.pipeline.to(torch_device)

    def get_init_params(self) -> Dict:
        """Return init params to determine what model is being used."""
        return {"model_name": self.model_name, "model_path": self.model_path}

    @torch.no_grad()
    def generate(
        self, prompt: Union[str, List[str]], **kwargs: Any
    ) -> List[Tuple[Any, float]]:
        """
        Generate the prompt from model.

        Outputs must be generated text and score, not including prompt.

        Args:
            prompt: promt to generate from.

        Returns:
            list of generated text (list of length 1 for 1 generation).
        """
        # TODO: Is this correct for getting arguments in?
        if isinstance(prompt, str):
            prompt = [prompt]
        result = self.pipeline(prompt, output_type="np.array", **kwargs)
        # Return None for logprobs
        return [(im, None) for im in result["images"]]

    @torch.no_grad()
    def logits_scoring(
        self, prompt: Union[str, List[str]], gold_choices: List[str], **kwargs: Any
    ) -> List[Tuple[Any, float]]:
        """
        Given the prompt and gold choices, choose the best choice with max logits.

        Args:
            prompt: promt to generate from.
            gold_choices: list of choices to choose from.

        Returns:
            the returned gold choice
        """
        raise NotImplementedError("Logits scoring not supported for diffusers")
