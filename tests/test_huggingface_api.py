"""Test the HuggingFace API."""

import math
import os
from subprocess import PIPE, Popen

import pytest

from manifest.api.models.huggingface import TextGenerationModel

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

    result = model.logits_scoring(inputs, gold_choices=[" blue sky", " green sky"])
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == " blue sky"
    assert math.isclose(round(result[0][1], 3), -6.999)

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

    result = model.logits_scoring(inputs, gold_choices=[" blue sky", " green sky"])
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == " green sky"
    assert math.isclose(round(result[0][1], 3), -13.538)

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
    assert math.isclose(round(result[0], 3), -19.935)
    assert math.isclose(round(result[1], 3), -45.831)


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

    result = model.logits_scoring(
        inputs, gold_choices=[" purple sky", " green sky", " blue sky"]
    )
    assert result is not None
    assert len(result) == 2
    assert result[0][0] == " blue sky"
    assert math.isclose(round(result[0][1], 3), -6.999)
    assert result[1][0] == " blue sky"
    assert math.isclose(round(result[1][1], 3), -8.212)

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

    result = model.logits_scoring(
        inputs, gold_choices=[" purple sky", " green sky", " blue sky"]
    )
    assert result is not None
    assert len(result) == 2
    assert result[0][0] == " green sky"
    assert math.isclose(round(result[0][1], 3), -13.538)
    assert result[1][0] == " blue sky"
    assert math.isclose(round(result[1][1], 3), -41.503) or math.isclose(
        round(result[1][1], 3), -41.504
    )

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
