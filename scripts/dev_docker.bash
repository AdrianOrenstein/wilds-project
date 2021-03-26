#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/home/wilds-project/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    -p 8888:8888 \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    wilds-project \
    bash 