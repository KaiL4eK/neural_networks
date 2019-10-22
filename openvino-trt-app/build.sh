#!/bin/bash

export TensorRT_ROOT="$HOME/TensorRT-5.0.2.6"

mkdir -p build; cd build;
cmake \
    .. && \

cmake --build . -- -j`nproc --all`
