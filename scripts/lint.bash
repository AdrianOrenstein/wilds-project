#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/app/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    --gpus all \
    --ipc host \
    wilds-project \
    /bin/bash -c " \
        black . && \
        isort . --settings-file=linters/isort.ini && \
        flake8  --config=linters/flake8.ini \
    "
    