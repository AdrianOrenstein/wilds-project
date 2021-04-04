#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/app/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    --gpus all \
    --ipc host \
    wilds-project \
    python3 src/download_wilds_datasets.py