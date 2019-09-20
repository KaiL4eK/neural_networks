#!/bin/bash

BASE_NET=SmallMobileNetv2_416x416_t1

./build/openvino-app -i 00382.ppm \
    -r ../TSD/src/_gen/uff_models/$BASE_NET.xml \
    -c ../TSD/src/_gen/uff_models/$BASE_NET.json
