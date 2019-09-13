#!/bin/bash

git clone https://github.com/opencv/dldt.git

docker run --rm -it --privileged \
    -v /dev:/dev -v `pwd`:/home/developer/ws \
    docker-openvino \
    bash -c \
    ". ~/.bashrc; cd dldt/inference-engine/samples; bash build_samples.sh && cd ~/inference_engine_samples_build/intel64/Release; ls; bash"
