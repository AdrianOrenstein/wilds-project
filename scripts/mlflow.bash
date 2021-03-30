#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/app/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    -p 5000:5000 \
    --gpus all \
    --ipc host \
    wilds-project \
    /bin/bash -c "mlflow ui --host 0.0.0.0"

