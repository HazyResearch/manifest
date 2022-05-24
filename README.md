# manifest
Prompt programming with FMs.

# Install
Download the code:
```
git clone git@github.com:HazyResearch/manifest.git
cd manifest
```

Install:
```
pip install poetry
poetry install
poetry run pre-commit install
```
or
```
pip install poetry
make dev
```

# Development
Before submitting a PR, run
```
export REDIS_PORT="6379"  # or whatever PORT local redis is running for those tests
make test
```
