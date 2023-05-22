0.1.9 - Unreleased
---------------------

0.1.8 - 2023-05-22
---------------------
Added
^^^^^
* Azure model support (completion and chat)
* Google Vertex API model support (completion and chat)
* Streaming responses for LM Completions (set stream=True)

Fixed
^^^^^
* `run` with batches now acts the same as async run except not async. We will batch requests into appropriate batchs sizes.
* Refactored client so unified preprocess and postprocess of requests and responses to better support model variants in request/response format.

0.1.7 - 2023-05-17
---------------------
Fixed
^^^^^
* `_run_chat` fixed bug where not passing in kwargs

0.1.6 - 2023-05-16
---------------------
Fixed
^^^^^
* Unified `run` and `run_chat` methods so it's just `run` now.
* LLama HF models for eval

0.1.5 - 2023-05-03
---------------------
Added
^^^^^
* Added chat input for chat models.

0.1.4 - 2023-04-24
---------------------
Added
^^^^^
* Connection pools to swap between clients
* Chunksize param for async runs

Fixed
^^^^^
* Determine cache and response by request type, not client name
* Refactor Response to use Pydantic types for Request and Response

0.1.1
---------------------
Added
^^^^^
* Async support in arun_batch

Fixed
^^^^^
* Batched runs now caches individual items
* Score prompt does not truncate outside token

Removed
^^^^^
* Deprecated chatGPT in favor of openaichat which uses OpenAI completions
* Deprecated Sessions

0.1.0 - 2022-01-31
---------------------
Added
^^^^^
* Batched inference support in `manifest.run`. No more separate `manifest.run_batch` method.
* Standard request base model for all language inputs.
* ChatGPT client. Requires CHATGPT_SESSION_KEY to be passed in.
* Diffusion model support
* Together model support

Removed
^^^^^^^
* `Prompt` class
* `OPT` client - OPT is now available in HuggingFace

0.0.1 - 2022-11-08
-------------------
First major pip release of Manifest. Install via `pip install manifest-ml`.


.. _@lorr1: https://github.com/lorr1
