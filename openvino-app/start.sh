#!/bin/bash

./build/openvino-app -d MYRIAD -i 00382.ppm \
    -r ../TSD/src/_gen/ir_models/TSD_SmallMbNv2_SmallMobileNetv2.xml \
    -c ../TSD/src/_gen/ir_models/TSD_SmallMbNv2_SmallMobileNetv2.json
