"""Huggingface model."""
import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    pipeline,
)

from manifest.api.models.model import Model

MODEL_REGISTRY = {
    "EleutherAI/gpt-j-6B": GPTJForCausalLM,
    "EleutherAI/gpt-neo-125M": GPTNeoForCausalLM,
    "EleutherAI/gpt-neo-1.3B": GPTNeoForCausalLM,
    "EleutherAI/gpt-neo-2.7B": GPTNeoForCausalLM,
    "gpt2": GPT2LMHeadModel,
    "bigscience/T0pp": AutoModelForSeq2SeqLM,
    "bigscience/T0_3B": AutoModelForSeq2SeqLM,
}

MODEL_PIPELINE = {
    "EleutherAI/gpt-j-6B": "text-generation",
    "EleutherAI/gpt-neo-125M": "text-generation",
    "EleutherAI/gpt-neo-1.3B": "text-generation",
    "EleutherAI/gpt-neo-2.7B": "text-generation",
    "gpt2": "text-generation",
    "bigscience/T0pp": "text2text-generation",
    "bigscience/T0_3B": "text2text-generation",
}


class HuggingFaceModel(Model):
    """Huggingface model."""

    def __init__(self, model_name: str, cache_dir: str, device: int):
        """
        Initialize model.

        All arguments will be passed in the request from Manifest.

        Args:
            model_name: model name string.
        """
        # Check if providing path
        self.model_path = model_name
        if Path(self.model_path).exists() and Path(self.model_path).is_dir():
            # Try to find config
            if (Path(self.model_path) / "config.json").exists():
                config = json.load(open(Path(self.model_path) / "config.json"))
                model_name = config["_name_or_path"]
        self.model_name = model_name
        print("Model Name:", self.model_name, "Model Path:", self.model_path)
        model = MODEL_REGISTRY[model_name].from_pretrained(
            self.model_path, cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline(
            MODEL_PIPELINE[model_name], model=model, tokenizer=tokenizer, device=device
        )
        self.returns_input = MODEL_PIPELINE[model_name] == "text-generation"

    def get_init_params(self) -> Dict:
        """Return init params to determine what model is being used."""
        return {"model_name": self.model_name, "model_path": self.model_path}

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
        final_results = []
        encoded_prompt = self.pipeline.tokenizer.encode(
            prompt, add_special_tokens=False
        )
        result = self.pipeline(
            prompt,
            max_length=kwargs.get("max_tokens") + len(encoded_prompt),
            temperature=kwargs.get("temperature"),
            repetition_penalty=kwargs.get("repetition_penalty"),
            top_k=kwargs.get("top_k"),
            top_p=kwargs.get("top_p"),
            num_return_sequences=num_return,
        )
        if self.returns_input:
            start_idx = len(prompt)
        else:
            start_idx = 0
        if num_return == 1:
            final_results.append(result[0]["generated_text"][start_idx:])
        else:
            final_results.append([r["generated_text"][start_idx:] for r in result])
        return final_results
