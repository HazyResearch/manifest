dev:
	pip install -e .[all]
	pre-commit install

test: dev check
	pytest tests

format:
	isort --atomic manifest/ tests/ web_app/
	black manifest/ tests/ web_app/

check:
	isort -c manifest/ tests/ web_app/
	black manifest/ tests/ web_app/ --check
	flake8 manifest/ tests/ web_app/
	mypy manifest/ tests/ web_app/

clean:
	pip uninstall -y manifest
	rm -rf src/manifest.egg-info
	rm -rf build/ dist/

prune:
	@bash -c "git fetch -p";
	@bash -c "for branch in $(git branch -vv | grep ': gone]' | awk '{print $1}'); do git branch -d $branch; done";

.PHONY: dev test clean check prune
