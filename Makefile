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

tests:
	bash scripts/tests.bash
	@echo "✅✅✅✅✅ Tests are good! ✅✅✅✅✅"

fake_experiment: lint tests
	bash scripts/fake_experiment.bash

