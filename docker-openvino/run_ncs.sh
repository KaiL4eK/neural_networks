#!/bin/bash

docker run --rm -it --privileged \
    -v /dev:/dev \
    docker-openvino
