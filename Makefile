PYTHON = python3
PIP = pip3

.DEFAULT_GOAL = run

build:
	bash scripts/build_docker.bash

run: build
	bash scripts/dev_docker.bash

stop:
	docker container kill $$(docker ps -q)

jupyter:
	bash scripts/jupyter.bash

lint:
	bash scripts/lint.bash
	@echo "✅✅✅✅✅ Lint is good! ✅✅✅✅✅"

test:
	bash scripts/test.bash
	@echo "✅✅✅✅✅ Tests are good! ✅✅✅✅✅"

test_experiment: lint test
	bash scripts/test_main.bash

