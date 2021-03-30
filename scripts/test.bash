#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/app/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    -p 8888:8888 \
    --gpus all \
    --ipc host \
    wilds-project \
    python3 tests/test.py
    