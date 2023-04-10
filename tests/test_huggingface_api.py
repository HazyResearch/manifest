"""Test the HuggingFace API."""

import math
import os
from subprocess import PIPE, Popen

import numpy as np
import pytest

from manifest.api.models.huggingface import MODEL_REGISTRY, TextGenerationModel
from manifest.api.models.sentence_transformer import SentenceTransformerModel

NOCUDA = 0
try:
    p = Popen(
        [
            "nvidia-smi",
            (
                "--query-gpu=index,utilization.gpu,memory.total,memory.used,"
                "memory.free,driver_version,name,gpu_serial,display_active,"
                "display_mode"
            ),
            "--format=csv,noheader,nounits",
        ],
        stdout=PIPE,
    )
except OSError:
    NOCUDA = 1

MAXGPU = 0
if NOCUDA == 0:
    try:
        p = os.popen(  # type: ignore
            "nvidia-smi --query-gpu=index --format=csv,noheader,nounits"
        )
        i = p.read().split("\n")  # type: ignore
        MAXGPU = int(i[-2]) + 1
    except OSError:
        NOCUDA = 1


def test_load_non_registry_model() -> None:
    """Test load model not in registry."""
    model_name = "NinedayWang/PolyCoder-160M"
    assert model_name not in MODEL_REGISTRY
    model = TextGenerationModel(
        model_name_or_path=model_name, model_type="text-generation"
    )
    result = model.generate("Why is the sky green?", max_tokens=10)
    assert result is not None


def test_gpt_generate() -> None:
    """Test pipeline generation from a gpt model."""
    model = TextGenerationModel(
        model_name_or_path="gpt2",
        use_accelerate=False,
        use_parallelize=False,
        use_bitsandbytes=False,
        use_deepspeed=False,
        use_fp16=False,
        device=-1,
    )
    inputs = "Why is the sky green?"
    result = model.generate(inputs, max_tokens=10)
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "\n\nThe sky is green.\n\nThe"
    assert math.isclose(round(result[0][1], 3), -11.516)

    result = model.generate("Cats are", max_tokens=10)
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == " not the only ones who are being targeted by the"
    assert math.isclose(round(result[0][1], 3), -21.069)

    result = model.generate(inputs, max_tokens=5)
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "\n\nThe sky is"
    assert math.isclose(round(result[0][1], 3), -6.046)

    # Truncate max length
    model.pipeline.max_length = 5
    result = model.generate(inputs, max_tokens=2)
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "\n\n"
    assert math.isclose(round(result[0][1], 3), -1.414)


def test_encdec_generate() -> None:
    """Test pipeline generation from a gpt model."""
    model = TextGenerationModel(
        model_name_or_path="google/t5-small-lm-adapt",
        use_accelerate=False,
        use_parallelize=False,
        use_bitsandbytes=False,
        use_deepspeed=False,
        use_fp16=False,
        device=-1,
    )
    inputs = "Why is the sky green?"
    result = model.generate(inputs, max_tokens=10)
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "What is the sky green? What is the sky"
    assert math.isclose(round(result[0][1], 3), -7.271)

    result = model.generate("Cats are", max_tokens=10)
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "a great way to get out of the house"
    assert math.isclose(round(result[0][1], 3), -13.868)

    result = model.generate(inputs, max_tokens=5)
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "What is the sky green"
    assert math.isclose(round(result[0][1], 3), -5.144)

    # Truncate max length
    model.pipeline.max_length = 5
    result = model.generate(inputs, max_tokens=2)
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "Is"
    assert math.isclose(round(result[0][1], 3), -4.233)


def test_gpt_score() -> None:
    """Test pipeline generation from a gpt model."""
    model = TextGenerationModel(
        model_name_or_path="gpt2",
        use_accelerate=False,
        use_parallelize=False,
        use_bitsandbytes=False,
        use_deepspeed=False,
        use_fp16=False,
        device=-1,
    )
    inputs = ["Why is the sky green?", "Cats are butterflies"]
    result = model.score_sequence(inputs)
    assert result is not None
    assert len(result) == 2
    assert math.isclose(round(result[0][0], 3), -46.71)
    assert math.isclose(round(result[1][0], 3), -12.752)
    assert isinstance(result[0][1], list)
    assert isinstance(result[1][1], list)


def test_embed() -> None:
    """Test embedding pipeline."""
    model = TextGenerationModel(
        model_name_or_path="gpt2",
        use_accelerate=False,
        use_parallelize=False,
        use_bitsandbytes=False,
        use_deepspeed=False,
        use_fp16=False,
        device=-1,
    )
    inputs = ["Why is the sky green?", "Cats are butterflies"]
    embeddings = model.embed(inputs)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 768)

    model2 = SentenceTransformerModel(
        model_name_or_path="all-mpnet-base-v2",
        use_accelerate=False,
        use_parallelize=False,
        use_bitsandbytes=False,
        use_deepspeed=False,
        use_fp16=False,
        device=-1,
    )
    inputs = ["Why is the sky green?", "Cats are butterflies"]
    embeddings = model2.embed(inputs)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 768)


def test_batch_gpt_generate() -> None:
    """Test pipeline generation from a gpt model."""
    model = TextGenerationModel(
        model_name_or_path="gpt2",
        use_accelerate=False,
        use_parallelize=False,
        use_bitsandbytes=False,
        use_deepspeed=False,
        use_fp16=False,
        device=-1,
    )
    inputs = ["Why is the sky green?", "Cats are"]
    result = model.generate(inputs, max_tokens=10)
    assert result is not None
    assert len(result) == 2
    assert result[0][0] == "\n\nThe sky is green.\n\nThe"
    assert math.isclose(round(result[0][1], 3), -11.516)
    assert result[1][0] == " not the only ones who are being targeted by the"
    assert math.isclose(round(result[1][1], 3), -21.069)

    result = model.generate(inputs, max_tokens=5)
    assert result is not None
    assert len(result) == 2
    assert result[0][0] == "\n\nThe sky is"
    assert math.isclose(round(result[0][1], 2), -6.05)
    assert result[1][0] == " not the only ones who"
    assert math.isclose(round(result[1][1], 3), -9.978)

    # Truncate max length
    model.pipeline.max_length = 5
    result = model.generate(inputs, max_tokens=2)
    assert result is not None
    assert len(result) == 2
    assert result[0][0] == "\n\n"
    assert math.isclose(round(result[0][1], 3), -1.414)
    assert result[1][0] == " not the"
    assert math.isclose(round(result[1][1], 3), -6.246)


def test_batch_encdec_generate() -> None:
    """Test pipeline generation from a gpt model."""
    model = TextGenerationModel(
        model_name_or_path="google/t5-small-lm-adapt",
        use_accelerate=False,
        use_parallelize=False,
        use_bitsandbytes=False,
        use_deepspeed=False,
        use_fp16=False,
        device=-1,
    )
    inputs = ["Why is the sky green?", "Cats are"]
    result = model.generate(inputs, max_tokens=10)
    assert result is not None
    assert len(result) == 2
    assert result[0][0] == "What is the sky green? What is the sky"
    assert math.isclose(round(result[0][1], 3), -7.271)
    assert result[1][0] == "a great way to get out of the house"
    assert math.isclose(round(result[1][1], 3), -13.868)

    result = model.generate(inputs, max_tokens=5)
    assert result is not None
    assert len(result) == 2
    assert result[0][0] == "What is the sky green"
    assert math.isclose(round(result[0][1], 3), -5.144)
    assert result[1][0] == "a great way to"
    assert math.isclose(round(result[1][1], 3), -6.353)

    # Truncate max length
    model.pipeline.max_length = 5
    result = model.generate(inputs, max_tokens=2)
    assert result is not None
    assert len(result) == 2
    assert result[0][0] == "Is"
    assert math.isclose(round(result[0][1], 3), -4.233)
    assert result[1][0] == "a"
    assert math.isclose(round(result[1][1], 3), -1.840)


@pytest.mark.skipif(
    (NOCUDA == 1 or MAXGPU == 0), reason="No cuda or GPUs found through nvidia-smi"
)
def test_gpt_deepspeed_generate() -> None:
    """Test deepspeed generation from a gpt model."""
    model = TextGenerationModel(
        model_name_or_path="gpt2",
        use_accelerate=False,
        use_parallelize=False,
        use_bitsandbytes=False,
        use_deepspeed=True,
        use_fp16=False,
        device=0,
    )
    inputs = "Why is the sky green?"
    result = model.generate(inputs, max_tokens=10)
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "\n\nThe sky is green.\n\nThe"
    assert math.isclose(round(result[0][1], 3), -11.517)
