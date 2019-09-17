#!/bin/bash

BASE_NET=TSD_SmallMbNv2_SmallMobileNetv2
# BASE_NET=TSD_SmallMbNv2Rect_SmallMobileNetv2

./build/openvino-app -d MYRIAD -i 00382.ppm \
    -r ../TSD/src/_gen/ir_models/$BASE_NET.xml \
    -c ../TSD/src/_gen/ir_models/$BASE_NET.json
