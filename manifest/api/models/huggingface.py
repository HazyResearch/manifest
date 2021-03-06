"""Huggingface model."""
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
    OPTForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from manifest.api.models.model import Model

MODEL_REGISTRY = {
    "EleutherAI/gpt-neo-1.3B": GPTNeoForCausalLM,
    "EleutherAI/gpt-neo-2.7B": GPTNeoForCausalLM,
    "EleutherAI/gpt-j-6B": GPTJForCausalLM,
    "EleutherAI/gpt-neox-20b": GPTNeoXForCausalLM,
    "facebook/opt-1.3b": OPTForCausalLM,
    "facebook/opt-2.7b": OPTForCausalLM,
    "facebook/opt-6.7b": OPTForCausalLM,
    "facebook/opt-13b": OPTForCausalLM,
    "facebook/opt-30b": OPTForCausalLM,
    "gpt2": GPT2LMHeadModel,
    "bigscience/T0pp": AutoModelForSeq2SeqLM,
    "bigscience/T0_3B": AutoModelForSeq2SeqLM,
}


class Pipeline:
    """
    Custom Pipeline.

    HF pipelines do not handle devices well in multi-gpu setting.
    Create our own generation pipeline.
    """

    def __init__(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: int
    ):
        """Initialize."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = (
            torch.device("cpu")
            if (device == -1 or not torch.cuda.is_available())
            else torch.device(f"cuda:{device}")
        )

    def __call__(self, text: str, **kwargs: Any) -> List[Dict[str, str]]:
        """Generate from text.

        Args:
            text: text to generate.

        Returns:
            generated text.
        """
        encoded_prompt = self.tokenizer.encode(text, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)
        output_sequences = self.model.generate(  # type: ignore
            encoded_prompt,
            max_length=kwargs.get("max_length"),
            temperature=kwargs.get("temperature"),
            top_k=kwargs.get("top_k"),
            top_p=kwargs.get("top_p"),
            repetition_penalty=kwargs.get("repetition_penalty"),
            do_sample=kwargs.get("do_sample"),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=kwargs.get("num_return_sequences"),
        )
        generated_sequences = [
            {
                "generated_text": self.tokenizer.decode(
                    output_seq, skip_special_tokens=True
                )
            }
            for output_seq in output_sequences
        ]
        return generated_sequences


class HuggingFaceModel(Model):
    """Huggingface model."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        device: int,
        use_accelerate: bool,
        perc_max_gpu_mem_red: float,
        use_fp32: bool,
    ):
        """
        Initialize model.

        All arguments will be passed in the request from Manifest.

        Args:
            model_name: model name string.
            cache_dir: cache directory for model.
            device: device to use for model.
            use_accelerate: whether to use accelerate for multi-gpu inference.
            perc_max_gpu_mem_red: percent max memory reduction in accelerate
            use_fp32: use fp32 for model weights.
        """
        # Check if providing path
        self.model_path = model_name
        if Path(self.model_path).exists() and Path(self.model_path).is_dir():
            # Try to find config
            if (Path(self.model_path) / "config.json").exists():
                config = json.load(open(Path(self.model_path) / "config.json"))
                model_name = config["_name_or_path"]
        self.model_name = model_name
        self.is_encdec = "T0" in self.model_name
        print("Model Name:", self.model_name, "Model Path:", self.model_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ValueError:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        dtype = torch.float32 if use_fp32 else torch.float16
        model = MODEL_REGISTRY[model_name].from_pretrained(  # type: ignore
            self.model_path, cache_dir=cache_dir, torch_dtype=dtype
        )
        if use_accelerate:
            self._dispatch_accelerate_model(model, perc_max_gpu_mem_red)
            device = 0
        else:
            if device > -1:
                model = model.to(device)  # type: ignore
        self.pipeline = Pipeline(  # type: ignore
            model=model, tokenizer=tokenizer, device=device
        )
        # Autogregressive models generate the input, too
        self.returns_input = not self.is_encdec

    def get_init_params(self) -> Dict:
        """Return init params to determine what model is being used."""
        return {"model_name": self.model_name, "model_path": self.model_path}

    def _dispatch_accelerate_model(
        self, model: PreTrainedModel, perc_max_gpu_mem_red: float
    ) -> None:
        """
        Load model with accelerate.

        Adapted from https://colab.research.google.com/drive/14wnxMvD9zsiBQo2FtT
                     pxn6w2cpXCcb-7#scrollTo=y8Ne7jJdaF9F&uniqifier=1

        Args:
            model: loaded hugging face model
            perc_max_gpu_mem_red: percent memory reduction
        """
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils.modeling import get_max_memory

        model.tie_weights()  # type: ignore
        # Get the model where we can infer devices from
        if hasattr(model, "model"):
            # OPT
            main_model = model.model  # type: ignore
            model_getter = "model."
        else:
            # Eleuther Neo and J
            main_model = model
            model_getter = ""
        # Decrease max mem
        max_memory = {
            k: int(perc_max_gpu_mem_red * v) for k, v in get_max_memory().items()
        }
        raw_device_map = infer_auto_device_map(
            main_model,
            max_memory=max_memory,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "GPTNeoBlock",
                "GPTJBlock",
                "GPTNeoXLayer",
                "T5Block",
            ],
            dtype=model.dtype,  # type: ignore
        )
        # Hacky fix for Eleuther getting the "weight" of embeddings
        device_map = {}
        for k, v in raw_device_map.items():
            if k in {"wte", "wpe"}:
                device_map[f"{model_getter}{k}.weight"] = v
            else:
                device_map[f"{model_getter}{k}"] = v
        # For OPT models
        if "lm_head" not in device_map:
            try:
                device_map["lm_head"] = max(device_map.values())
            except TypeError:
                device_map["lm_head"] = "cpu"
        print("Device Map", device_map)
        dispatch_model(model, device_map=device_map)
        return

    def generate(self, prompt: str, **kwargs: Any) -> List[str]:
        """
        Generate the prompt from model.

        Outputs must be generated text, not including prompt.

        Args:
            prompt: promt to generate from.

        Returns:
            list of generated text (list of length 1 for 1 generation).
        """
        num_return = kwargs.get("n")
        # Add tokens for length
        encoded_prompt_with_special = self.pipeline.tokenizer.encode(prompt)
        # Remove tokens as the pipeline removes special tokens upon return
        encoded_prompt_without_special = self.pipeline.tokenizer.encode(
            prompt, add_special_tokens=False
        )
        result = self.pipeline(
            prompt,
            max_length=kwargs.get("max_tokens") + len(encoded_prompt_with_special),
            temperature=kwargs.get("temperature"),
            repetition_penalty=kwargs.get("repetition_penalty"),
            top_k=kwargs.get("top_k"),
            top_p=kwargs.get("top_p"),
            do_sample=kwargs.get("do_sample"),
            num_return_sequences=num_return,
        )
        # Correctly returns prompt without extra spaces
        decoded_prompt = self.pipeline.tokenizer.decode(encoded_prompt_without_special)
        if self.returns_input:
            start_idx = len(decoded_prompt)
        else:
            start_idx = 0
        if num_return == 1:
            final_results = [result[0]["generated_text"][start_idx:]]
        else:
            final_results = [r["generated_text"][start_idx:] for r in result]
        return final_results
