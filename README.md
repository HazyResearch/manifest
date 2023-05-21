# Manifest
How to make prompt programming with Foundation Models a little easier.


# Table of Contents
- [Install](#install)
- [Getting Started](#getting-started)
- [Manifest](#manifest-components)
- [Other Models Types](#other-models)
    - [Local HuggingFace Models](#local-huggingface-models)
    - [Chat Models](#chat-models)
    - [Embedding Models](#embedding-models)
- [Road Map](#road-map)
- [Development](#development)
- [Cite](#cite)


# Install
Install:
```bash
pip install manifest-ml
```

Install with diffusion support:
```bash
pip install manifest-ml[diffusers]
```

Install with HuggingFace local model support:
```bash
pip install manifest-ml[api]
```

Dev Install:
```bash
git clone git@github.com:HazyResearch/manifest.git
cd manifest
make dev
```

# Getting Started
Running is simple to get started. If using OpenAI, set `export OPENAI_API_KEY=<OPENAIKEY>` (or pass key in through variable `client_connection`) then run

```python
from manifest import Manifest

# Start a manifest session to OpenAI - default `engine=text-davinci-003`
manifest = Manifest(
    client_name = "openai",
)
manifest.run("Why is the grass green?")
```

## Examples
We have example notebook and python scripts located at [examples](examples). These show how to use different models, model types (i.e. text, diffusers, or embedding models), and async running.

# Manifest Components
Manifest is meant to be a very light weight package to help with prompt design and iteration. Three key design decisions of Manifest are

* All models are behind APIs
* Supports caching of model inputs/outputs for iteration, reproducibility, and cost saving
* Unified API to support generate, score, and embed

## Models
Manifest provides model clients for [OpenAI](https://openai.com/), [AI21](https://studio.ai21.com/), [Cohere](https://cohere.ai/), [Together](https://together.xyz/), and HuggingFace (see [below](#huggingface-models) for how to use locally hosted HuggingFace models). You can toggle between the models by changing `client_name` and `client_connection`. For example, if a HuggingFace model is loaded locally, run
```python
manifest = Manifest(
    client_name = "huggingface",
    client_connection = "http://127.0.0.1:5000",
)
```
If you want to use Cohere, run
```python
manifest = Manifest(
    client_name = "cohere",
    client_connection = <COHERE_API_KEY>,
)
```
You can also just set `export COHERE_API_KEY=<COHERE_API_KEY>` and not use `client_connection`.


You can see the model details and possible model inputs to `run()` via
```python
print(manifest.client_pool.get_current_client().get_model_params())
print(manifest.client_pool.get_current_client().get_model_inputs())
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

## Running Queries
Once you have a session open, you can write and develop prompts.

```python
result = manifest.run("Hello, my name is Laurel")
```

You can also run over multiple examples if supported by the client.
```python
results = manifest.run(["Where are the cats?", "Where are the dogs?"])
```

We support async queries as well via
```python
import asyncio
results = asyncio.run(manifest.arun_batch(["Where are the cats?", "Where are the dogs?"]))
```

If something doesn't go right, you can also ask to get a raw manifest Response.
```python
result_object = manifest.run(["Where are the cats?", "Where are the dogs?"], return_response=True)
print(result_object.get_request_obj())
print(result_object.is_cached())
print(result_object.get_response_obj())
```

By default, we do not truncate results based on a stop token. You can change this by either passing a new stop token to a Manifest session or to a `run`.
```python
result = manifest.run(prompt, "Laurel", stop_token="and")
```

If you want to change default parameters to a model, we pass those as `kwargs` to the client.
```python
result = manifest.run(prompt, "Laurel", max_tokens=50)
```

## Streaming Queries
Manifest also supports streaming the model response back, assuming it's supported by the underlying client. When calling `run`, pass `stream=True` to get a streaming iterator in response.

```python
result_iterator = manifest.run("Tell me a story. Once upon a time", max_tokens=100, stream=True)
for res_text in result_iterator:
    print(res_text)
```
Streaming responses are only supported for single string queries (not batch mode) for text completion models.

## Model Pools
Manifest supports querying multiple models with different schedulers. This is very much a work in progress effort, but Manifest will round robin select (or randomly select) the clients you want. You can use the same client multiple times with different connection strings (e.g. different API keys), or you can mix and match. The only requirement is that all clients are the same request type. I.e. you can't have a pool of generation models and embedding models.

To query between a local model and OpenAI,
```python
from manifest.connections.client_pool import ClientConnection
from manifest import Manifest

client_connection1 = ClientConnection(
    client_name="huggingface",
    client_connection="http://127.0.0.1:5000",
)
client_connection2 = ClientConnection(client_name="openai", engine="text-ada-001")
manifest = Manifest(
    client_pool=[client_connection1, client_connection2],
    cache_name="sqlite",
    client_connection=sqlite_cache,
)
manifest.run(...)
```

The speed benefit comes in with async batched runs. When calling `arun_batch` with a list of prompts, Manifest supports a `chunk_size` param. This will break the prompts into `chunk_size` chunks to spread across the client pool. By default `chunk_size` is `-1` which means only one client will get all the prompts to run asynchronously. You must set `chunk_size > 1` to distribute across the pool. There is a further `batch_size` param which control the individual client `batch_size` to send to the model.

```python
responses = asyncio.run(manifest.arun_batch(prompts, max_tokens=30, chunk_size=20))
```

# Other Models

## Local Huggingface Models
To use a HuggingFace generative model, in `manifest/api` we have a Flask application that hosts the models for you.

In a separate terminal or Tmux/Screen session, to load 6B parameters models, run
```bash
python3 -m manifest.api.app \
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

To help load larger models, we also support using `parallelize()` from HF, [accelerate](https://huggingface.co/docs/accelerate/index), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), and [deepspeed](https://github.com/microsoft/DeepSpeed). You will need to install these packages first via `pip install manifest-ml[api]`. We list the commands to load larger models below.

* T0pp
```bash
python3 -m manifest.api.app \
    --model_type huggingface \
    --model_name_or_path bigscience/T0pp \
    --use_hf_parallelize
```

* NeoX 20B (requires at least 60GB of GPU memory)
```bash
python3 -m manifest.api.app \
    --model_type huggingface \
    --model_name_or_path EleutherAI/gpt-neox-20b \
    --use_accelerate_multigpu \
    --percent_max_gpu_mem_reduction 0.75
```
* Bloom 175B (requires at least 240GB of GPU memory)
```bash
python3 -m manifest.api.app \
    --model_type huggingface \
    --model_name_or_path bigscience/bloom \
    --use_bitsandbytes \
    --percent_max_gpu_mem_reduction 0.85
```

## Chat Models
Manifest has specific support for executing against chat models in the more standard "system" / "user" dialogue. To pass in a dialogue history to Manifest, use the `run` command with a list of dictionary inputs with `role` and `content` keys using an associated chat model such as `openaichat`.

```python
manifest = Manifest(client_name="openaichat")
dialogue = [
    {"role": "system", "content": "You are a helpful assistant who also responds in rhymes"},
    {"role": "user", "content": "What is the date?"},
]
res = manifest.run(dialogue, max_tokens=100)
```

## Embedding Models
Manifest also supports getting embeddings from models and available APIs. We do this all through changing the `client_name` argument. You still use `run` and `abatch_run`.

To use OpenAI's embedding models, simply run
```python
manifest = Manifest(client_name="openaiembedding")
embedding_as_np = manifest.run("Get me an embedding for a bunny")
```

As explained above, you can load local HuggingFace models that give you embeddings, too. If you want to use a standard generative model, load the model as above use use `client_name="huggingfaceembedding"`. If you want to use a standard embedding model, like those from SentenceTransformers, load your local model via
```bash
python3 -m manifest.api.app \
    --model_type sentence_transformers \
    --model_name_or_path all-mpnet-base-v2 \
    --device 0
```

# Road Map
Here's what's coming up next
- [ ] Clients
  - [ ] HuggingFace Hub
  - [x] Azure OpenAI
  - [x] Google Vertex
  - [ ] Anthropic
  - [x] Streaming Support Completions
  - [ ] Streaming Support Chat Models
- [ ] Data Types
  - [ ] Diffusion Models
- [x] Orchestration
  - [x] Connection pools
- [ ] Local Inference
  - [ ] FlexGen

# Development
Before submitting a PR, run
```bash
export REDIS_PORT="6379"  # or whatever PORT local redis is running for those tests
cd <REDIS_PATH>
docker run -d -p 127.0.0.1:${REDIS_PORT}:6379 -v `pwd`:`pwd` -w `pwd` --name manifest_redis_test redis
make test
```

# Cite
Please cite Manifest if you used it for any publications. Thanks!!
```
@misc{orr2022manifest,
  author = {Orr, Laurel},
  title = {Manifest},
  year = {2022},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/HazyResearch/manifest}},
}
```
