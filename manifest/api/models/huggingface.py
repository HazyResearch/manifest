"""Huggingface model."""
from typing import Any, List

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
}

MODEL_PIPELINE = {
    "EleutherAI/gpt-j-6B": "text-generation",
    "EleutherAI/gpt-neo-125M": "text-generation",
    "EleutherAI/gpt-neo-1.3B": "text-generation",
    "EleutherAI/gpt-neo-2.7B": "text-generation",
    "gpt2": "text-generation",
    "bigscience/T0pp": "text2text-generation",
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
        model = MODEL_REGISTRY[model_name].from_pretrained(
            model_name, cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline(
            MODEL_PIPELINE[model_name], model=model, tokenizer=tokenizer, device=device
        )

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
        # Removes tokens removed from tokenization
        decoded_prompt = self.pipeline.tokenizer.decode(
            encoded_prompt, clean_up_tokenization_spaces=True
        )
        if num_return == 1:
            final_results.append(result[0]["generated_text"][len(decoded_prompt) :])
        else:
            final_results.append(
                [r["generated_text"][len(decoded_prompt) :] for r in result]
            )
        return final_results
