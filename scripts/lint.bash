#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/app/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    -p 8888:8888 \
    --gpus all \
    --ipc host \
    wilds-project \
    /bin/bash -c " \
        black . && \
        isort . --settings-file=linters/isort.ini && \
        flake8  --config=linters/flake8.ini \
    "
    