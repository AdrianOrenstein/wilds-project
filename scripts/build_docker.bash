#!/usr/bin/env bash

docker build -t wilds-project \
  -f dockerfiles/Dockerfile \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  .