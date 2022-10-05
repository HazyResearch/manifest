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
        config = model.config  # type: ignore
        # Used for GPT
        self.max_length = getattr(config, "max_position_embeddings", None)
        if self.max_length is None:
            # Used for T0
            self.max_length = config.d_model
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
        # If text is longer than max model length, we reduce max input length to ensure
        # the user indicated generation tokens is preserved.
        max_input_length = kwargs.get("max_input_length")
        encoded_prompt = self.tokenizer.encode(
            text, max_length=max_input_length, truncation=True, return_tensors="pt"
        )
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
        use_parallelize: bool,
        perc_max_gpu_mem_red: float,
        use_fp16: bool,
    ):
        """
        Initialize model.

        All arguments will be passed in the request from Manifest.

        Args:
            model_name: model name string.
            cache_dir: cache directory for model.
            device: device to use for model.
            use_accelerate: whether to use accelerate for multi-gpu inference.
            use_parallelize: use HF default parallelize
            perc_max_gpu_mem_red: percent max memory reduction in accelerate
            use_fp16: use fp16 for model weights.
        """
        if use_accelerate and use_parallelize:
            raise ValueError("Cannot use both accelerate and parallelize")
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

        if not self.is_encdec:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if use_fp16 else "auto"
        try:
            # Try to explicitely find a fp16 copy (gpt-j-6B for example)
            model = MODEL_REGISTRY[model_name].from_pretrained(  # type: ignore
                self.model_path,
                cache_dir=cache_dir,
                revision="float16",
                torch_dtype=torch.float16,
            )
        except Exception:
            model = MODEL_REGISTRY[model_name].from_pretrained(  # type: ignore
                self.model_path, cache_dir=cache_dir, torch_dtype=dtype
            )
        model.eval()
        print(f"Loaded Model DType {model.dtype}")
        if use_accelerate:
            self._dispatch_accelerate_model(model, perc_max_gpu_mem_red)
            device = 0
        elif use_parallelize:
            model.parallelize()
            device = 0
        else:
            if device > -1:
                torch_device = (
                    torch.device("cpu")
                    if (device == -1 or not torch.cuda.is_available())
                    else torch.device(f"cuda:{device}")
                )
                model = model.to(torch_device)  # type: ignore
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
        max_input_len = self.pipeline.max_length - kwargs.get("max_tokens")
        # Add tokens for length
        encoded_prompt_with_special = self.pipeline.tokenizer.encode(
            prompt, max_length=max_input_len, truncation=True
        )
        # Remove tokens as the pipeline removes special tokens upon return
        encoded_prompt_without_special = self.pipeline.tokenizer.encode(
            prompt, max_length=max_input_len, truncation=True, add_special_tokens=False
        )
        result = self.pipeline(
            prompt,
            max_input_length=max_input_len,
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

    def logits_scoring(
        self, prompt: str, gold_choices: List[str], **kwargs: Any
    ) -> str:
        """
        Given the prompt and gold choices, choose the best choice with max logits.

        Args:
            prompt: promt to generate from.
            gold_choices: list of choices to choose from.

        Returns:
            the returned gold choice
        """
        max_input_len = self.pipeline.max_length
        # Adapted from https://github.com/bigscience-workshop/t-zero
        tokenized_inputs = self.pipeline.tokenizer(
            prompt,
            padding="longest",
            max_length=max_input_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Get max target length
        max_target_len = max(
            [
                len(self.pipeline.tokenizer(ans_choi)["input_ids"])
                for ans_choi in gold_choices
            ]
        )
        tokenized_targets = [
            self.pipeline.tokenizer(
                ans_choi,
                # padding is on the right here.
                padding="max_length",
                max_length=min(max_target_len, max_input_len),
                truncation=True,
            )
            for ans_choi in gold_choices
        ]

        # Repeat input ids for each choice to form a batch
        features = {
            k: [tokenized_inputs[k] for _ in range(len(gold_choices))]
            for k in tokenized_inputs.keys()
        }
        # Add choice tokens + mask
        features["labels"] = [
            tokenized_targets[k]["input_ids"] for k in range(len(gold_choices))
        ]
        features["labels_attention_mask"] = [
            tokenized_targets[k]["attention_mask"] for k in range(len(gold_choices))
        ]
        # Convert to tensors
        tensor_features = {}
        for k in features:
            tensor_features[k] = torch.LongTensor(features[k]).to(self.pipeline.device)
        # Reduce GPU memory by feeding one at a time
        logits = [
            self.pipeline.model(  # type: ignore
                input_ids=tensor_features["input_ids"][bs].unsqueeze(0),
                attention_mask=tensor_features["attention_mask"][bs].unsqueeze(0),
                labels=tensor_features["labels"][bs].unsqueeze(0),
            ).logits
            for bs in range(len(tensor_features["input_ids"]))
        ]
        stacked_logits = torch.vstack(logits)
        # Compute most likely option
        masked_log_probs = tensor_features["labels_attention_mask"].unsqueeze(
            -1
        ) * torch.log_softmax(stacked_logits, dim=-1)
        seq_token_log_probs = torch.gather(
            masked_log_probs, -1, tensor_features["labels"].unsqueeze(-1)
        )
        seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
        prediction = seq_log_prob.argmax(dim=-1).item()
        return gold_choices[int(prediction)]
