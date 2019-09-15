#!/bin/bash

./build/openvino-app -d MYRIAD -i 00382.ppm \
    -r ../openvino/models/TSD_SmallMbNv2_SmallMobileNetv2.xml \
    -c ../openvino/models/small-mbn-v2-tiles.json
