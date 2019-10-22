#!/bin/bash

export TensorRT_ROOT="$HOME/TensorRT-5.0.2.6"

mkdir -p build; cd build;
cmake \
    -D TensorRT_SAMPLES_INCLUDE_DIR="/usr/src/tensorrt/samples/common" \
    .. && \

cmake --build . -- -j`nproc --all`
