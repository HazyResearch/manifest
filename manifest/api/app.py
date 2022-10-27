"""Flask app."""
import argparse
import logging
import os
import socket
from typing import Dict

import pkg_resources
from flask import Flask, request

from manifest.api.models.huggingface import HuggingFaceModel
from manifest.api.response import Response

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
app = Flask(__name__)  # define app using Flask
# Will be global
model = None
PORT = int(os.environ.get("FLASK_PORT", 5000))
MODEL_CONSTRUCTORS = {
    "huggingface": HuggingFaceModel,
}
try:
    from manifest.api.models.zoo import ZooModel

    MODEL_CONSTRUCTORS["zoo"] = ZooModel  # type: ignore
except ImportError:
    logger.warning("Zoo model not available.")


def parse_args() -> argparse.Namespace:
    """Generate args."""
    parser = argparse.ArgumentParser(description="Model args")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type used for finding constructor.",
        choices=["huggingface", "zoo"],
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Name of model or path to model. Used in initialize of model class.",
    )
    parser.add_argument(
        "--model_config",
        default=None,
        type=str,
        help="Model config. Used in initialize of model class.",
    )
    parser.add_argument(
        "--cache_dir", default=None, type=str, help="Cache directory for models."
    )
    parser.add_argument(
        "--device", type=int, default=-1, help="Model device. -1 for CPU."
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
    args = parser.parse_args()
    return args


def is_port_in_use(port: int) -> bool:
    """Check if port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def main() -> None:
    """Run main."""
    kwargs = parse_args()
    if is_port_in_use(PORT):
        raise ValueError(f"Port {PORT} is already in use.")

    model_type = kwargs.model_type
    model_name_or_path = kwargs.model_name_or_path
    model_config = kwargs.model_config
    if not model_name_or_path and not model_config:
        raise ValueError("Must provide model_name_or_path or model_config.")
    use_accelerate = kwargs.use_accelerate_multigpu
    if use_accelerate:
        logger.info("Using accelerate. Overridding --device argument.")
    if (
        kwargs.percent_max_gpu_mem_reduction <= 0
        or kwargs.percent_max_gpu_mem_reduction > 1
    ):
        raise ValueError("percent_max_gpu_mem_reduction must be in (0, 1].")
    # Global model
    global model
    model = MODEL_CONSTRUCTORS[model_type](
        model_name_or_path,
        model_config=model_config,
        cache_dir=kwargs.cache_dir,
        device=kwargs.device,
        use_accelerate=use_accelerate,
        use_parallelize=kwargs.use_hf_parallelize,
        use_bitsandbytes=kwargs.use_bitsandbytes,
        perc_max_gpu_mem_red=kwargs.percent_max_gpu_mem_reduction,
        use_fp16=kwargs.fp16,
    )
    app.run(host="0.0.0.0", port=PORT)


@app.route("/completions", methods=["POST"])
def completions() -> Dict:
    """Get completions for generation."""
    prompt = request.json["prompt"]
    del request.json["prompt"]
    generation_args = request.json

    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a str")

    results_text = []
    for generations in model.generate(prompt, **generation_args):
        results_text.append(generations)
    results = [{"text": r[0], "text_logprob": r[1]} for r in results_text]
    # transform the result into the openai format
    return Response(results, response_type="text_completion").__dict__()


@app.route("/choice_logits", methods=["POST"])
def choice_logits() -> Dict:
    """Get maximal likely choice via max logits after generation."""
    prompt = request.json["prompt"]
    del request.json["prompt"]
    gold_choices = request.json["gold_choices"]
    del request.json["gold_choices"]
    generation_args = request.json

    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a str")

    if not isinstance(gold_choices, list):
        raise ValueError("Gold choices must be a list of string choices")

    result, score = model.logits_scoring(prompt, gold_choices, **generation_args)
    results = [{"text": result, "text_logprob": score}]
    # transform the result into the openai format
    return Response(results, response_type="choice_selection").__dict__()


@app.route("/params", methods=["POST"])
def params() -> Dict:
    """Get model params."""
    return model.get_init_params()


@app.route("/")
def index() -> str:
    """Get index completion."""
    fn = pkg_resources.resource_filename("metaseq", "service/index.html")
    with open(fn) as f:
        return f.read()


if __name__ == "__main__":
    main()
