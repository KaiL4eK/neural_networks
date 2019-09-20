#!/bin/bash

mkdir -p build; cd build;
cmake -D TensorRT_ROOT="$HOME/TensorRT-6.0.1.5" .. && cmake --build . -- -j`nproc --all`
