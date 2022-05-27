dev:
	poetry install
	poetry run pre-commit install
	poetry run mypy --install-types

test: dev check
	poetry run pytest tests

format:
	poetry run isort --atomic manifest/ tests/
	poetry run black manifest/ tests/

check:
	poetry run isort -c manifest/ tests/
	poetry run black manifest/ tests/ --check
	poetry run flake8 manifest/ tests/
	poetry run mypy manifest/

clean:
	pip uninstall -y manifest
	rm -rf src/manifest.egg-info
	rm -rf build/ dist/

prune:
	@bash -c "git fetch -p";
	@bash -c "for branch in $(git branch -vv | grep ': gone]' | awk '{print $1}'); do git branch -d $branch; done";

.PHONY: dev test clean check prune
