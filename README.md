# Manifest
How to make prompt programming with Foundation Models a little easier.


# Table of Contents
- [Install](#install)
- [Getting Started](#getting-started)
- [Manifest](#manifest-components)
- [Local HuggingFace Models](#local-huggingface-models)
- [Development](#development)


# Install
Download the code:
```bash
git clone git@github.com:HazyResearch/manifest.git
cd manifest
```

Install:
```bash
pip install -e .
```

Dev Install:
```bash
make dev
```

# Getting Started
Running is simple to get started. If using OpenAI, set `export OPENAI_API_KEY=<OPENAIKEY>` then run

```python
from manifest import Manifest

# Start a manifest session to OpenAI - default `engine=text-davinci-002`
manifest = Manifest(
    client_name = "openai",
)
manifest.run("Why is the grass green?")
```

# Manifest Components
Manifest is meant to be a very light weight package to help with prompt design and iteration. Three key design decisions of Manifest are

* Prompt are functional -- they can take an input example and dynamically change
* All models are behind APIs
* Supports caching of model inputs/outputs for iteration, reproducibility, and cost saving

## Models
Manifest provides model clients for OpenAI, AI21, OPT (assuming model is loaded locally), and HuggingFace (see [below](#huggingface-models) for how to use locally hosted HuggingFace models). You can toggle between the models by changing `client_name` and `client_connection`. For example, if a HuggingFace model is loaded locally, run
```python
manifest = Manifest(
    client_name = "huggingface",
    client_connection = "http://127.0.0.1:5000",
)
```

You can see the model details and possible model inputs to `run()` via
```python
print(manifest.client.get_model_params())
print(manifest.client.get_model_inputs())
```

## Prompts
A Manifest prompt is a function that accepts a single input to generate a string prompt to send to a model.

```python
from manifest import Prompt
prompt = Prompt(lambda x: f"Hello, my name is {x}")
print(prompt("Laurel"))
>>> "Hello, my name is Laurel"
```

Running
```python
result = manifest.run(prompt, "Laurel")
```
will send ``Hello, my name is Laurel'' to the model.

As you saw above, if you don't want your prompt to change, we also support static strings
```python
result = manifest.run("Hello, my name is static")
```

## Global Cache
We support having queries and results stored in a global cache that can be shared across users. We treat inputs and outputs as key value pairs and support SQLite or Redis backends. To start with global caching using SQLite, run

```python
manifest = Manifest(
    client_name = "openai",
    cache_name = "sqlite",
    cache_connection = "mycache.sqlite",
)
```
The cache will be saved in `mycache.sqlite`.

We also support Redis backend.
```python
manifest = Manifest(
    client_name = "openai",
    cache_name = "redis",
    cache_connection = "localhost:6379"
)
```
As a hint, if you want to get Redis running, see the `docker run` command below under development.

## Sessions
Each Manifest run supports a session that, in addition to a global cache, connects to a local SQLite DB to store user query history.
```python
manifest = Manifest(
    client_name = "openai",
    cache_name = "sqlite",
    cache_connection = "mycache.sqlite",
    session_id = "grass_color",
)
```
will start a Manifest session with the session name `grass_color`. This can be helpful for a user to logically keep track of sessions, see interaction history, and resume sessions if desired. If the session id provided is `_default`, we generate a random id for the user.

After a few queries, the user can explore their history
```python
manifest.get_last_queries(4)
```
will retrieve the last 4 model queries and responses.

## Running Queries
Once you have a session open, you can write and develop prompts.

```python
prompt = Prompt(lambda x: "Hello, my name is {x}")
result = manifest.run(prompt, "Laurel")
```

You can also run over multiple examples.
```python
results = manifest.run_batch(prompt, ["Laurel", "Avanika"])
```

If something doesn't go right, you can also ask to get a raw manifest Response.
```python
result_objects = manifest.batch_run(prompt, ["Laurel", "Avanika"], return_response=True)
for result_object in result_objects:
    print(result_object.get_request())
    print(result_object.is_cached())
    print(result_object.get_json_response())
```

By default, we do not truncate results based on a stop token. You can change this by either passing a new stop token to a Manifest session or to a `run` or `run_batch`.
```python
result = manifest.run(prompt, "Laurel", stop_token="and")
```

If you want to change default parameters to a model, we pass those as `kwargs` to the client.
```python
result = manifest.run(prompt, "Laurel", max_tokens=50)
```

# Local Huggingface Models
To use a HuggingFace generative model, in `manifest/api` we have a Falsk application that hosts the models for you.

In a separate terminal or Tmux/Screen session, to load 6B parameters models, run
```bash
python3 manifest/api/app.py \
    --model_type huggingface \
    --model_name_or_path EleutherAI/gpt-j-6B \
    --device 0
```
You will see the Flask session start and output a URL `http://127.0.0.1:5000`. Pass this in to Manifest. If you want to use a different port, set the `FLASK_PORT` environment variable.

```python
manifest = Manifest(
    client_name = "huggingface",
    client_connection = "http://127.0.0.1:5000",
)
```

If you have a custom model you trained, pass the model path to `--model_name_or_path`.

To help load larger models, we also support using `parallelize()` from HF, [accelerate](https://huggingface.co/docs/accelerate/index), and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). You will need to install these packages first. We list the commands to load larger models below.

* T0pp
```bash
python3 manifest/api/app.py \
    --model_type huggingface \
    --model_name_or_path bigscience/T0pp \
    --use_hf_parallelize
```

* NeoX 20B (requires at least 60GB of GPU memory)
```bash
python3 manifest/api/app.py \
    --model_type huggingface \
    --model_name_or_path EleutherAI/gpt-neox-20b \
    --use_accelerate_multigpu \
    --percent_max_gpu_mem_reduction 0.75
```
* Bloom 175B (requires at least 240GB of GPU memory)
```bash
python3 manifest/api/app.py \
    --model_type huggingface \
    --model_name_or_path bigscience/bloom \
    --use_bitsandbytes \
    --percent_max_gpu_mem_reduction 0.85
```

# Development
Before submitting a PR, run
```bash
export REDIS_PORT="6380"  # or whatever PORT local redis is running for those tests
cd <REDIS_PATH>
docker run -d -p 127.0.0.1:${REDIS_PORT}:6379 -v `pwd`:`pwd` -w `pwd` --name manifest_redis_test redis
make test
```
