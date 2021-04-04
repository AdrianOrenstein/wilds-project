PYTHON = python3
PIP = pip3

.DEFAULT_GOAL = run

build:
	bash scripts/build_docker.bash

run: build
	bash scripts/dev_docker.bash

stop:
	docker container kill $$(docker ps -q)

jupyter: build
	bash scripts/jupyter.bash

lint: build
	bash scripts/lint.bash
	@echo "✅✅✅✅✅ Lint is good! ✅✅✅✅✅"

tests: build
	bash scripts/tests.bash
	@echo "✅✅✅✅✅ Tests are good! ✅✅✅✅✅"

mlflow: build
	bash scripts/mlflow.bash

download: build
	bash scripts/download_wilds_datasets.bash

fake_experiment: build lint tests
	bash scripts/fake_experiment.bash

