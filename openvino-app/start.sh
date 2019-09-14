#!/bin/bash

./build/openvino-app -d VPU -i 00382.ppm \
    -r ../openvino/models/TSD_SmallMbNv2_100_Tiles_SmallMobileNetv2.xml \
    -c ../TSD/src/cfgs/small-mbn-v2-tiles.json 
