0.1.6 - Unreleased
---------------------

0.1.5 - 2022-05-03
---------------------
Added
^^^^^
* Added chat input for chat models.

0.1.4 - 2022-04-24
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
