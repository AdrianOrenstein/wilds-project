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
	bash -i scripts/jupyter.bash

lint:
	black . 
	isort . --settings-file=project/linters/isort.ini
	flake8  --config=project/linters/flake8.ini
	@echo "✅✅✅✅✅ Good to go! ✅✅✅✅✅"

