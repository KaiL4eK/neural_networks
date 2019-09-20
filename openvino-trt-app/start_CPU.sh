#!/bin/bash

BASE_NET=SmallMobileNetv2_416x416_t1
# BASE_NET=SmallMobileNetv2_416x416_t2

./build/openvino-app -d CPU -i 00382.ppm \
    -r ../TSD/src/_gen/ir_models/"$BASE_NET"_FP32.xml \
    -c ../TSD/src/_gen/ir_models/$BASE_NET.json
