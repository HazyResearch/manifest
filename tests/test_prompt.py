"""Prompt test."""
import pytest

from manifest import Prompt


def test_init():
    """Test prompt initialization."""
    str_prompt = "This is a test prompt"
    func_prompt = lambda: "This is a test prompt"
    func_single_prompt = lambda x: f"{x} is a test prompt"
    func_list_prompt = lambda x: f"{x[0]} is a test {x[1]}"
    func_double_prompt = lambda x, y: f"{x} is a test {y}"
    # TODO: add list of prompt tests

    # String prompt
    prompt = Prompt(str_prompt)
    assert prompt(None) == str_prompt
    assert prompt() == str_prompt

    # Function no inputs
    prompt = Prompt(func_prompt)
    assert prompt(None) == str_prompt
    assert prompt() == str_prompt

    # Function single inputs
    prompt = Prompt(func_single_prompt)
    assert prompt("This") == str_prompt
    assert prompt("Hello") == "Hello is a test prompt"

    # Function list inputs
    prompt = Prompt(func_list_prompt)
    assert prompt(["This", "prompt"]) == str_prompt
    assert prompt(["Hello", "prompt"]) == "Hello is a test prompt"

    # Function two inputs
    with pytest.raises(ValueError) as exc_info:
        Prompt(func_double_prompt)
    assert str(exc_info.value) == "Prompts must have zero or one input."


def test_serialize():
    """Test prompt serialization."""
    str_prompt = "This is a test prompt"
    func_single_prompt = lambda x: f"{x} is a test prompt"

    # String prompt
    prompt = Prompt(str_prompt)
    assert Prompt.deserialize(prompt.serialize()).prompt_func() == prompt.prompt_func()

    # Function single inputs
    prompt = Prompt(func_single_prompt)
    assert Prompt.deserialize(prompt.serialize()).prompt_func(1) == prompt.prompt_func(
        1
    )
