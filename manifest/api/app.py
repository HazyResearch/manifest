"""Flask app."""
import argparse
import io
import json
import logging
import os
import socket
from typing import Dict

from flask import Flask, Response, request

from manifest.api.models.diffuser import DiffuserModel
from manifest.api.models.huggingface import (
    MODEL_GENTYPE_REGISTRY,
    CrossModalEncoderModel,
    TextGenerationModel,
)
from manifest.api.models.sentence_transformer import SentenceTransformerModel
from manifest.api.response import ModelResponse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
app = Flask(__name__)  # define app using Flask
# Will be global
model = None
model_type = None
PORT = int(os.environ.get("FLASK_PORT", 5000))
MODEL_CONSTRUCTORS = {
    "huggingface": TextGenerationModel,
    "sentence_transformers": SentenceTransformerModel,
    "huggingface_crossmodal": CrossModalEncoderModel,
    "diffuser": DiffuserModel,
}


def parse_args() -> argparse.Namespace:
    """Generate args."""
    parser = argparse.ArgumentParser(description="Model args")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type used for finding constructor.",
        choices=MODEL_CONSTRUCTORS.keys(),
    )
    parser.add_argument(
        "--model_generation_type",
        default=None,
        type=str,
        help="Model generation type.",
        choices=MODEL_GENTYPE_REGISTRY.keys(),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Name of model or path to model. Used in initialize of model class.",
    )
    parser.add_argument(
        "--cache_dir", default=None, type=str, help="Cache directory for models."
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Model device. -1 for CPU."
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Force use fp16 for model params."
    )
    parser.add_argument(
        "--percent_max_gpu_mem_reduction",
        type=float,
        default=0.85,
        help="Used with accelerate multigpu. Scales down max memory.",
    )
    parser.add_argument(
        "--use_bitsandbytes",
        action="store_true",
        help=("Use bits and bytes. " "This will override --device parameter."),
    )
    parser.add_argument(
        "--use_accelerate_multigpu",
        action="store_true",
        help=(
            "Use accelerate for multi gpu inference. "
            "This will override --device parameter."
        ),
    )
    parser.add_argument(
        "--use_hf_parallelize",
        action="store_true",
        help=(
            "Use HF parallelize for multi gpu inference. "
            "This will override --device parameter."
        ),
    )
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help=("Use deepspeed. This will override --device parameter."),
    )
    parser.add_argument(
        "--is_flask_debug",
        action="store_true",
        help=("If TRUE, then run Flask in debug mode with autoreload."),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=("Specify the port to run Flask server on. Defaults to FLASK_PORT environment variable. If that's not set, defaults to 5000"),
    )
    args = parser.parse_args()
    return args


def is_port_in_use(port: int) -> bool:
    """Check if port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def main() -> None:
    """Run main."""
    kwargs = parse_args()
    # if is_port_in_use(PORT):
    #     raise ValueError(f"Port {PORT} is already in use.")
    global model_type
    model_type = kwargs.model_type
    model_gen_type = kwargs.model_generation_type
    model_name_or_path = kwargs.model_name_or_path
    if not model_name_or_path:
        raise ValueError("Must provide model_name_or_path.")
    if kwargs.use_accelerate_multigpu:
        logger.info("Using accelerate. Overridding --device argument.")
    if (
        kwargs.percent_max_gpu_mem_reduction <= 0
        or kwargs.percent_max_gpu_mem_reduction > 1
    ):
        raise ValueError("percent_max_gpu_mem_reduction must be in (0, 1].")
    if (
        sum(
            [
                kwargs.use_accelerate_multigpu,
                kwargs.use_hf_parallelize,
                kwargs.use_bitsandbytes,
                kwargs.use_deepspeed,
            ]
        )
        > 1
    ):
        raise ValueError(
            "Only one of use_accelerate_multigpu, use_hf_parallelize, "
            "use_bitsandbytes, and use_deepspeed can be set."
        )
    # Global model
    global model
    model = MODEL_CONSTRUCTORS[model_type](
        model_name_or_path,
        model_type=model_gen_type,
        cache_dir=kwargs.cache_dir,
        device=kwargs.device,
        use_accelerate=kwargs.use_accelerate_multigpu,
        use_parallelize=kwargs.use_hf_parallelize,
        use_bitsandbytes=kwargs.use_bitsandbytes,
        use_deepspeed=kwargs.use_deepspeed,
        perc_max_gpu_mem_red=kwargs.percent_max_gpu_mem_reduction,
        use_fp16=kwargs.fp16,
    )
    app.run(host="0.0.0.0", port=kwargs.port, debug=kwargs.is_flask_debug)


@app.route("/completions", methods=["POST"])
def completions() -> Response:
    """Get completions for generation."""
    prompt = request.json["prompt"]
    del request.json["prompt"]
    generation_args = request.json

    if not isinstance(prompt, (str, list)):
        raise ValueError("Prompt must be a str or list of str")
    try:
        result_gens = []
        for generations in model.generate(prompt, **generation_args):
            result_gens.append(generations)
        if model_type == "diffuser":
            # Assign None logprob as it's not supported in diffusers
            results = [
                {"array": r[0], "logprob": None, "tokens": None, "token_logprobs": None}
                for r in result_gens
            ]
            res_type = "image_generation"
        else:
            results = [
                {"text": r[0], "logprob": r[1], "tokens": r[2], "token_logprobs": r[3]}
                for r in result_gens
            ]
            res_type = "text_completion"
        # transform the result into the openai format
        return Response(
            json.dumps(ModelResponse(results, response_type=res_type).__dict__()),
            status=200,
        )
    except Exception as e:
        logger.error(e)
        import traceback
        print(traceback.format_exc())
        return Response(
            json.dumps({"message": str(e)}),
            status=400,
        )


@app.route("/embed", methods=["POST"])
def embed() -> Response:
    """Get embed for generation."""
    if "modality" in request.json:
        modality = request.json["modality"]
    else:
        modality = "text"
    if modality == "text":
        prompts = request.json["prompt"]
    elif modality == "image":
        import base64

        from PIL import Image

        prompts = [
            Image.open(io.BytesIO(base64.b64decode(data)))
            for data in request.json["prompt"]
        ]
    else:
        raise ValueError("modality must be text or image")

    try:
        results = []
        embeddings = model.embed(prompts)
        for embedding in embeddings:
            results.append(
                {
                    "array": embedding,
                    "logprob": None,
                    "tokens": None,
                    "token_logprobs": None,
                }
            )

        return Response(
            json.dumps(
                ModelResponse(results, response_type="embedding_generation").__dict__()
            ),
            status=200,
        )
    except Exception as e:
        logger.error(e)
        return Response(
            json.dumps({"message": str(e)}),
            status=400,
        )


@app.route("/score_sequence", methods=["POST"])
def score_sequence() -> Response:
    """Get logprob of prompt."""
    prompt = request.json["prompt"]
    del request.json["prompt"]
    generation_args = request.json

    if not isinstance(prompt, (str, list)):
        raise ValueError("Prompt must be a str or list of str")

    try:
        score_list = model.score_sequence(prompt, **generation_args)
        results = [
            {
                "text": prompt if isinstance(prompt, str) else prompt[i],
                "logprob": r[0],
                "tokens": r[1],
                "token_logprobs": r[2],
            }
            for i, r in enumerate(score_list)
        ]
        # transform the result into the openai format
        return Response(
            json.dumps(
                ModelResponse(results, response_type="prompt_logit_score").__dict__()
            ),
            status=200,
        )
    except Exception as e:
        logger.error(e)
        return Response(
            json.dumps({"message": str(e)}),
            status=400,
        )


@app.route("/score_sequence_eleuther_lm_eval", methods=["POST"])
def score_sequence_eleuther_lm_eval() -> Response:
    """Get logprob of prompt."""
    prompts_with_labels = request.json["prompts_with_labels"]
    del request.json["prompts_with_labels"]
    generation_args = request.json

    if not isinstance(prompts_with_labels, (tuple, list)):
        raise ValueError("Prompt must be a tuple or list of tuples")

    if isinstance(prompts_with_labels, tuple):
        prompts_with_labels = [prompts_with_labels]

    try:
        score_list = model.score_sequence_eleuther_lm_eval(prompts_with_labels, **generation_args)
        results = [
            {
                "prompt": p,
                "label" : l,
                "label_prob": probs,
            }
            for p, l, probs in score_list
        ]
        # transform the result into the openai format
        return Response(json.dumps(results), status=200)
    except Exception as e:
        logger.error(e)
        return Response(
            json.dumps({"message": str(e)}),
            status=400,
        )

@app.route("/params", methods=["POST"])
def params() -> Dict:
    """Get model params."""
    return model.get_init_params()


@app.route("/model_config", methods=["GET"])
def model_config() -> Dict:
    """Get model config."""
    return model.pipeline.model.config.to_dict()

@app.route("/tokenizer_config", methods=["GET"])
def tokenizer_config() -> Dict:
    """Get tokenizer config."""
    return model.pipeline.tokenizer.__repr__()

@app.route("/tokenize", methods=["POST"])
def tokenize() -> Dict:
    """Get tokenized version of prompt."""
    prompt = request.json["prompt"]
    encoded_prompt = model.tokenize(prompt)
    return {
        'input_ids' : encoded_prompt['input_ids'],
        'attention_mask' : encoded_prompt['attention_mask'],
    }


@app.route("/")
def index() -> str:
    """Get index completion."""
    return 'Welcome to Manifest. Try using a different endpoint.'


if __name__ == "__main__":
    main()
