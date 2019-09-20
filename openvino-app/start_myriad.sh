#!/bin/bash

BASE_NET=SmallMobileNetv2_416x416_t1

./build/openvino-app -d MYRIAD -i 00382.ppm \
    -r ../TSD/src/_gen/ir_models/"$BASE_NET"_FP16.xml \
    -c ../TSD/src/_gen/ir_models/$BASE_NET.json
