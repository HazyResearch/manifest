"""Setup for all tests."""
import os
import shutil
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import redis

from manifest.request import DiffusionRequest, EmbeddingRequest, LMRequest
from manifest.response import ArrayModelChoice, LMModelChoice, ModelChoices


@pytest.fixture
def model_choice() -> ModelChoices:
    """Get dummy model choice."""
    model_choices = ModelChoices(
        choices=[
            LMModelChoice(
                text="hello", token_logprobs=[0.1, 0.2], tokens=["hel", "lo"]
            ),
            LMModelChoice(text="bye", token_logprobs=[0.3], tokens=["bye"]),
        ]
    )
    return model_choices


@pytest.fixture
def model_choice_single() -> ModelChoices:
    """Get dummy model choice."""
    model_choices = ModelChoices(
        choices=[
            LMModelChoice(
                text="helloo", token_logprobs=[0.1, 0.2], tokens=["hel", "loo"]
            ),
        ]
    )
    return model_choices


@pytest.fixture
def model_choice_arr() -> ModelChoices:
    """Get dummy model choice."""
    np.random.seed(0)
    model_choices = ModelChoices(
        choices=[
            ArrayModelChoice(array=np.random.randn(4, 4), token_logprobs=[0.1, 0.2]),
            ArrayModelChoice(array=np.random.randn(4, 4), token_logprobs=[0.3]),
        ]
    )
    return model_choices


@pytest.fixture
def model_choice_arr_int() -> ModelChoices:
    """Get dummy model choice."""
    np.random.seed(0)
    model_choices = ModelChoices(
        choices=[
            ArrayModelChoice(
                array=np.random.randint(20, size=(4, 4)), token_logprobs=[0.1, 0.2]
            ),
            ArrayModelChoice(
                array=np.random.randint(20, size=(4, 4)), token_logprobs=[0.3]
            ),
        ]
    )
    return model_choices


@pytest.fixture
def request_lm() -> LMRequest:
    """Get dummy request."""
    request = LMRequest(prompt=["what", "cat"])
    return request


@pytest.fixture
def request_lm_single() -> LMRequest:
    """Get dummy request."""
    request = LMRequest(prompt="monkey", engine="dummy")
    return request


@pytest.fixture
def request_array() -> EmbeddingRequest:
    """Get dummy request."""
    request = EmbeddingRequest(prompt="hello")
    return request


@pytest.fixture
def request_diff() -> DiffusionRequest:
    """Get dummy request."""
    request = DiffusionRequest(prompt="hello")
    return request


@pytest.fixture
def sqlite_cache(tmp_path: Path) -> Generator[str, None, None]:
    """Sqlite Cache."""
    cache = str(tmp_path / "sqlite_cache.sqlite")
    yield cache
    shutil.rmtree(cache, ignore_errors=True)


@pytest.fixture
def redis_cache() -> Generator[str, None, None]:
    """Redis cache."""
    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", 6379))
    yield f"{host}:{port}"
    # Clear out the database
    try:
        db = redis.Redis(host=host, port=port)
        db.flushdb()
    # For better local testing, pass if redis DB not started
    except redis.exceptions.ConnectionError:
        pass


@pytest.fixture
def postgres_cache(monkeypatch: pytest.MonkeyPatch) -> Generator[str, None, None]:
    """Postgres cache."""
    import sqlalchemy  # type: ignore

    # Replace the sqlalchemy.create_engine function with a function that returns an
    # in-memory SQLite engine
    url = sqlalchemy.engine.url.URL.create("sqlite", database=":memory:")
    engine = sqlalchemy.create_engine(url)
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *args, **kwargs: engine)
    return engine  # type: ignore
