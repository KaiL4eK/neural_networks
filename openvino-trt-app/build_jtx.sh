#!/bin/bash

mkdir -p build; cd build;
cmake \
    -D TensorRT_ROOT="$HOME/TensorRT-6.0.1.5" \
    -D TensorRT_SAMPLES_INCLUDE_DIR="/usr/src/tensorrt/samples/common" \
    .. && \

cmake --build . -- -j`nproc --all`
