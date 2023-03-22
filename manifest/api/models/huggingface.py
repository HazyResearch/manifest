"""Huggingface model."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import PIL
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils.modeling import get_max_memory as acc_get_max_memory
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BloomForCausalLM,
    CLIPModel,
    CLIPProcessor,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
    OPTForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)

import deepspeed
from manifest.api.models.model import Model

MODEL_REGISTRY = {
    "EleutherAI/gpt-neo-125M": GPTNeoForCausalLM,
    "EleutherAI/gpt-neo-1.3B": GPTNeoForCausalLM,
    "EleutherAI/gpt-neo-2.7B": GPTNeoForCausalLM,
    "EleutherAI/gpt-j-6B": GPTJForCausalLM,
    "EleutherAI/gpt-neox-20b": GPTNeoXForCausalLM,
    "facebook/opt-125m": OPTForCausalLM,
    "facebook/opt-350m": OPTForCausalLM,
    "Salesforce/codegen-2B-mono": AutoModelForCausalLM,
    "Salesforce/codegen-6B-mono": AutoModelForCausalLM,
    "facebook/opt-1.3b": OPTForCausalLM,
    "facebook/opt-2.7b": OPTForCausalLM,
    "facebook/opt-6.7b": OPTForCausalLM,
    "facebook/opt-13b": OPTForCausalLM,
    "facebook/opt-30b": OPTForCausalLM,
    "gpt2": GPT2LMHeadModel,
    "openai/clip-vit-base-patch32": CLIPModel,
    "bigscience/bloom-560m": BloomForCausalLM,
    "bigscience/bloom-1b7": BloomForCausalLM,
    "bigscience/bloom-3b": BloomForCausalLM,
    "bigscience/bloom-7b1": BloomForCausalLM,
    "bigscience/bloom": AutoModelForCausalLM,
    "bigscience/T0pp": AutoModelForSeq2SeqLM,
    "bigscience/T0_3B": AutoModelForSeq2SeqLM,
    "google/t5-small-lm-adapt": AutoModelForSeq2SeqLM,  # 220M
    "google/t5-l-lm-adapt": AutoModelForSeq2SeqLM,  # 800M
    "google/t5-xl-lm-adapt": AutoModelForSeq2SeqLM,  # 3B
    "google/t5-xxl-lm-adapt": AutoModelForSeq2SeqLM,  # 11B
    "google/t5-v1_1-l": AutoModelForSeq2SeqLM,  # 800M
    "google/t5-v1_1-xl": AutoModelForSeq2SeqLM,  # 3B
    "google/t5-v1_1-xxl": AutoModelForSeq2SeqLM,  # 11B
    "google/flan-t5-l": AutoModelForSeq2SeqLM,  # 800M
    "google/flan-t5-xl": AutoModelForSeq2SeqLM,  # 3B
    "google/flan-t5-xxl": AutoModelForSeq2SeqLM,  # 11B
}

MODEL_GENTYPE_REGISTRY = {
    "text-generation": AutoModelForCausalLM,
    "text2text-generation": AutoModelForSeq2SeqLM,
}


def get_max_memory(gpu_reduction: float) -> Dict[int, str]:
    """Get max memory in GB times reduction."""
    free_in_gb = int(torch.cuda.mem_get_info()[0] / 1024**3)  # type: ignore
    max_mem = f"{int(gpu_reduction*free_in_gb)}GB"

    n_gpus = torch.cuda.device_count()
    max_mem_dict = {i: max_mem for i in range(n_gpus)}
    return max_mem_dict


class GenerationPipeline:
    """
    Custom Pipeline.

    HF pipelines do not handle devices well in multi-gpu setting.
    Create our own generation pipeline.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, deepspeed.InferenceEngine],
        tokenizer: PreTrainedTokenizer,
        device: int = None,
        bitsandbytes: bool = False,
        is_encdec: bool = False,
    ):
        """Initialize."""
        # Use to turn off sampling
        # https://github.com/TimDettmers/bitsandbytes/issues/42
        self.bitsandbytes = bitsandbytes
        self.model = model
        self.is_encdec = is_encdec
        config = model.config  # type: ignore
        # Used for GPT
        self.max_length = getattr(config, "max_position_embeddings", None)
        if self.max_length is None:
            # Used for Bloom
            self.max_length = getattr(config, "seq_length", None)
            if self.max_length is None:
                # Used for T0
                self.max_length = getattr(config, "d_model", None)
                if self.max_length is None:
                    # Default
                    self.max_length = 2048

        print(f"Usings max_length: {self.max_length}")

        self.tokenizer = tokenizer
        # self.device = device
        # With bits and bytes, do not want to place inputs on any device
        # if self.device:
        self.device = (
            torch.device("cpu")
            if (device == -1 or not torch.cuda.is_available())
            else torch.device(f"cuda:{device}")
        )

    def __call__(
        self, text: Union[str, List[str]], **kwargs: Any
    ) -> List[Dict[str, Union[str, List[float]]]]:
        """Generate from text.

        Args:
            text: text to generate.

        Returns:
            generated text.
        """
        # set generation params
        max_new_tokens = kwargs.get("max_new_tokens", 30)
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", 50)
        top_p = kwargs.get("top_p", 1)
        repetition_penalty = kwargs.get("repetition_penalty", 1)
        do_sample = kwargs.get("do_sample", False)
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        print(f"Generating with parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}, do_sample={do_sample}, num_return_sequences={num_return_sequences}")
        
        # If text is longer than max model length, we reduce max input length to ensure
        # the user indicated generation tokens is preserved.
        max_input_len = (
            self.max_length - max_new_tokens
            if not self.is_encdec
            else self.max_length
        )
        encoded_prompt = self.tokenizer(
            text,
            max_length=max_input_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded_prompt = encoded_prompt.to(self.device)
        
        output_dict = self.model.generate(  # type: ignore
            input_ids=encoded_prompt['input_ids'],
            attention_mask=encoded_prompt['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample if not self.bitsandbytes else False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
        )
        # logits/scores from the output always correspond to the generated tokens.
        # shape (num_tokens, num_return_sequences, vocab_size)
        logits = torch.stack(output_dict.scores)
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        num_generated_tokens = logits.shape[0]
        generated_sequences = [
            {
                "generated_text": self.tokenizer.decode(
                    output_seq[-num_generated_tokens:], skip_special_tokens=True
                ),
                "logprobs": logits[
                    range(num_generated_tokens), i, output_seq[-num_generated_tokens:]
                ].tolist(),
            }
            for i, output_seq in enumerate(output_dict.sequences)
        ]
        return generated_sequences


class HuggingFaceModel(Model):
    """HuggingFace Model."""

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
        if sum([use_accelerate, use_parallelize, use_bitsandbytes, use_deepspeed]) > 1:
            raise ValueError(
                "Only one of use_accelerate, use_parallelize, "
                "use_bitsandbytes, use_deepspeed can be set to True"
            )
        # Check if providing path
        self.model_path = model_name_or_path
        # if Path(self.model_path).exists() and Path(self.model_path).is_dir():
        #     # Try to find config
        #     if (Path(self.model_path) / "config.json").exists():
        #         config = json.load(open(Path(self.model_path) / "config.json"))
        #         model_name_or_path = config["_name_or_path"]
        self.model_name = model_name_or_path
        self.model_type = model_type
        if self.model_name not in MODEL_REGISTRY and self.model_type is None:
            raise ValueError(
                f"{self.model_name} is not in our registry. Please specify "
                "--model_generation_type as either text-generation (for Causal)"
                " or text2text-generation (for Seq2Seq)"
            )
        print("Model Name:", self.model_name, "Model Path:", self.model_path)

    def get_init_params(self) -> Dict:
        """Return init params to determine what model is being used."""
        return {"model_name": self.model_name, "model_path": self.model_path}

    def _dispatch_deepspeed_model(
        self, model: PreTrainedModel
    ) -> deepspeed.InferenceEngine:
        """
        Load model with deepspeed.

        Adapted from https://www.deepspeed.ai/tutorials/inference-tutorial/

        Args:
            model: loaded hugging face model
        """
        model = deepspeed.init_inference(
            model=model,
            mp_size=1,
            dtype=model.dtype,
            replace_method="auto",
            replace_with_kernel_inject=True,
        )
        return model

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
            k: int(perc_max_gpu_mem_red * v) for k, v in acc_get_max_memory().items()
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

class CrossModalEncoderModel(HuggingFaceModel):
    """CrossModalEncoderModel."""

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
        super().__init__(
            model_name_or_path,
            model_type,
            model_config,
            cache_dir,
            device,
            use_accelerate,
            use_parallelize,
            use_bitsandbytes,
            use_deepspeed,
            perc_max_gpu_mem_red,
            use_fp16,
        )

        # TODO: make this generalizable
        self.processor = CLIPProcessor.from_pretrained(self.model_path)

        model = MODEL_REGISTRY.get(
            self.model_name, MODEL_GENTYPE_REGISTRY.get(self.model_type, None)
        ).from_pretrained(
            self.model_path,
            cache_dir=cache_dir,
        )
        model.eval()

        torch_device = (
            torch.device("cpu")
            if (device == -1 or not torch.cuda.is_available())
            else torch.device(f"cuda:{device}")
        )
        self.model = model.to(torch_device)  # type: ignore

    @torch.no_grad()
    def embed(self, prompt: Union[str, List[str]], **kwargs: Any) -> np.ndarray:
        """
        Compute embedding for prompts.

        Args:
            prompt: promt to generate from.

        Returns:
            embedding
        """
        if isinstance(prompt, str):
            inputs = self.processor(text=prompt, return_tensors="pt", padding=True)
        elif isinstance(prompt, PIL.Image.Image):
            inputs = self.processor(images=prompt, return_tensors="pt", padding=True)
        else:
            raise ValueError("Prompt must be a string or an image")

        outputs = self.model(**inputs)
        return outputs


class TextGenerationModel(HuggingFaceModel):
    """Huggingface model."""

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
        super().__init__(
            model_name_or_path,
            model_type,
            model_config,
            cache_dir,
            device,
            use_accelerate,
            use_parallelize,
            use_bitsandbytes,
            use_deepspeed,
            perc_max_gpu_mem_red,
            use_fp16,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, truncation_side="left", padding_side="left",
            )
        except ValueError:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                truncation_side="left",
                padding_side="left",
                use_fast=False,
            )
        dtype = torch.float16 if use_fp16 else "auto"
        if use_bitsandbytes:
            print("WARNING!!! Cannot use sampling with bitsandbytes.")
            max_memory = get_max_memory(perc_max_gpu_mem_red)
            model = MODEL_REGISTRY.get(
                self.model_name, MODEL_GENTYPE_REGISTRY.get(self.model_type, None)
            ).from_pretrained(  # type: ignore
                self.model_path,
                cache_dir=cache_dir,
                load_in_8bit=True,
                device_map="auto",
                max_memory=max_memory,
            )
        else:
            model = None
            if use_fp16:
                try:
                    # Try to find an explicit float16 model
                    model = MODEL_REGISTRY.get(
                        self.model_name, MODEL_GENTYPE_REGISTRY.get(self.model_type, None)
                    ).from_pretrained(  # type: ignore
                        self.model_path, cache_dir=cache_dir, revision="float16", torch_dtype=dtype,
                    )
                except:
                    # Couldn't find explicit float16 model
                    pass
            if model is None:
                model = MODEL_REGISTRY.get(
                    self.model_name, MODEL_GENTYPE_REGISTRY.get(self.model_type, None)
                ).from_pretrained(  # type: ignore
                    self.model_path, cache_dir=cache_dir, torch_dtype=dtype
                )
        model.eval()
        print(f"Loaded Model DType {model.dtype}")

        self.is_encdec = model.config.is_encoder_decoder
        if not self.is_encdec:
            tokenizer.pad_token = tokenizer.eos_token

        if not use_bitsandbytes:
            if use_accelerate:
                self._dispatch_accelerate_model(model, perc_max_gpu_mem_red)
                device = 0
            elif use_parallelize:
                model.parallelize()
                device = 0
            elif use_deepspeed:
                self._dispatch_deepspeed_model(model)
                device = 0
            else:
                if device > -1:
                    torch_device = (
                        torch.device("cpu")
                        if (device == -1 or not torch.cuda.is_available())
                        else torch.device(f"cuda:{device}")
                    )
                    model = model.to(torch_device)  # type: ignore
        self.pipeline = GenerationPipeline(  # type: ignore
            model=model,
            tokenizer=tokenizer,
            device=device,
            bitsandbytes=use_bitsandbytes,
            is_encdec=self.is_encdec,
        )

    @torch.no_grad()
    def embed(self, prompt: Union[str, List[str]], **kwargs: Any) -> np.ndarray:
        """
        Compute embedding for prompts.

        Args:
            prompt: promt to generate from.

        Returns:
            embedding
        """
        pass

    @torch.no_grad()
    def generate(
        self, prompt: Union[str, List[str]], **kwargs: Any
    ) -> List[Tuple[Any, float, List[float]]]:
        """
        Generate the prompt from model.

        Outputs must be generated text and score, not including prompt.

        Args:
            prompt: promt to generate from.

        Returns:
            list of generated text (list of length 1 for 1 generation).
        """
        num_return = kwargs.get("n", 1)
        if isinstance(prompt, list) and num_return > 1:
            raise ValueError("In batch generate, n must be 1.")
        result = self.pipeline(
            prompt,
            max_new_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            repetition_penalty=kwargs.get("repetition_penalty"),
            top_k=kwargs.get("top_k"),
            top_p=kwargs.get("top_p"),
            do_sample=kwargs.get("do_sample"),
            num_return_sequences=num_return,
        )
        final_results = [
            (
                cast(str, r["generated_text"]),
                sum(cast(List[float], r["logprobs"])),
                cast(List[float], r["logprobs"]),
            )
            for r in result
        ]
        return final_results
    
    @torch.no_grad()
    def score_sequence(
        self, prompt: Union[str, List[str]], **kwargs: Any
    ) -> List[Tuple[float, List[float]]]:
        """
        Score a sequence of choices.
        Args:
            prompt (:obj:`str` or :obj:`List[str]`):
                The prompt to score the choices against.
            **kwargs:
                Additional keyword arguments passed along to the :obj:`__call__` method.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        encoded_prompt = self.pipeline.tokenizer(
            prompt,
            max_length=self.pipeline.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded_prompt["labels"] = encoded_prompt["input_ids"].clone()
        encoded_prompt = encoded_prompt.to(self.pipeline.device)
        logits = self.pipeline.model(  # type: ignore
            **encoded_prompt,
        ).logits
        # For causal decoders, shift logts and labels
        labels_attention_mask = encoded_prompt["attention_mask"].unsqueeze(-1)[
            ..., 1:, :
        ]
        masked_log_probs = (
            labels_attention_mask.float()
            * torch.log_softmax(logits.float(), dim=-1)[..., :-1, :]
        )
        seq_token_log_probs = torch.gather(
            masked_log_probs, -1, encoded_prompt["labels"][..., 1:].unsqueeze(-1)
        )
        seq_token_log_probs = seq_token_log_probs.squeeze(dim=-1)
        seq_log_prob = seq_token_log_probs.sum(dim=-1)
        return [
            (seq, seq_token)
            for seq, seq_token in zip(
                seq_log_prob.tolist(), seq_token_log_probs.tolist()
            )
        ]

    def tokenize(self, prompt: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """Tokenize input."""
        if isinstance(prompt, str):
            prompt = [prompt]
        encoded_prompt = self.pipeline.tokenizer(
            prompt,
            max_length=self.pipeline.max_length,
            truncation=kwargs.get('truncation', True),
            padding=kwargs.get('padding', True),
            **kwargs,
        )
        return encoded_prompt

    @torch.no_grad()
    def score_sequence_eleuther_lm_eval(
        self, prompt_with_label: List[Tuple[str,str]], **kwargs: Any
    ) -> List[Tuple[str, str, float]]:
        """
        Score a sequence of (prompt + label).

        Args:
            prompt_with_label: List[Tuple[str,str]]
                The prompt to score the labels against, plus its corresponding label.
                Unlike the other score_sequence method, the label is not included in the prompt.
            **kwargs:
                Additional keyword arguments passed along to the :obj:`__call__` method.
        """

        prompts: List[str] = [ x[0] for x in prompt_with_label ]
        labels: List[str] = [ x[1] for x in prompt_with_label ]
        
        if self.model_type == 'text2text-generation':
            # For seq2seq models, we add the label as the output expected from the decoder, and
            # separate the label from the prompt (input to the encoder)
            encoded_prompt = self.pipeline.tokenizer(
                prompts,
                max_length=self.pipeline.max_length,
                truncation=True,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            encoded_prompt["labels"] = self.pipeline.tokenizer(
                                            labels, 
                                            max_length=self.pipeline.max_length,
                                            return_tensors="pt",
                                            truncation=True,
                                            padding=True,
                                            add_special_tokens=False,
                                        )['input_ids']
            encoded_prompt = encoded_prompt.to(self.pipeline.device)
            # Generate model logits for `labels`
            outputs = self.pipeline.model(  # type: ignore
                **encoded_prompt
            )
            log_softmaxes = torch.log_softmax(outputs.logits, dim=-1)
            label_mask = encoded_prompt["labels"] != self.pipeline.tokenizer.pad_token_id
            label_token_probs = torch.gather(log_softmaxes, -1, encoded_prompt["labels"].unsqueeze(-1)).squeeze(-1)
            label_probs = (label_mask * label_token_probs).sum(dim=-1)
        elif self.model_type == 'text-generation':
            # For gpt2-like models, we append the label to the end of the prompt, measure
            # the entire sequence's logprob, and count that as the label's logprob (since 
            # the label is the only thing that changes between sequences, so the prompt's logprob
            # will be constant across all labels)
            
            # Pad right instead of left
            encoded_prompt = self.pipeline.tokenizer(
                [ f"{x} {y}" for (x,y) in zip(prompts, labels) ],
                max_length=self.pipeline.max_length,
                truncation=True,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            encoded_prompt = encoded_prompt.to(self.pipeline.device)
            # Generate model logits for `labels`
            outputs = self.pipeline.model(  # type: ignore
                **encoded_prompt
            )
            prompt_lens: List[int] = [ len(x) for x in self.pipeline.tokenizer(
                [ f"{x}" for (x,y) in zip(prompts, labels) ],
                max_length=self.pipeline.max_length,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )['input_ids'] ]
            log_softmaxes = torch.log_softmax(outputs.logits, dim=-1)
            label_mask = torch.ones(log_softmaxes.shape[0], log_softmaxes.shape[1])
            for idx, prompt_len in enumerate(prompt_lens):
                # Mask out the prompt (plus any initial left padding) so that we're left with
                # just the tokens corresponding to the label
                left_padding_in_encoded_prompt: int = (encoded_prompt['attention_mask'][idx, :] == 0).sum().item()
                label_mask[idx, :prompt_len + left_padding_in_encoded_prompt] = 0
            # Shift everything to the left by one token, so that we're measuring the logprob of the *next* token
            shifted_input_ids = torch.roll(encoded_prompt['input_ids'], -1, dims=1).unsqueeze(-1)
            shifted_label_mask = torch.roll(label_mask, -1, dims=1).to(self.pipeline.device)
            # Calculate probability of next token at each position
            all_token_probs = torch.gather(log_softmaxes, -1, shifted_input_ids).squeeze(-1)
            # Drop everything except the logprobs corresponding to the labels
            label_probs = (shifted_label_mask * all_token_probs).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return list(zip(prompts, labels, label_probs.tolist()))