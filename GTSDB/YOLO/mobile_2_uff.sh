#!/bin/bash

python export_2_pb.py -c cfgs/mobile_config.json -w chk/mobile_bestMap.h5 -o mobile && \
convert-to-uff tensorflow -l --input-file output/mobile.pb && \
python tensorrt_inf.py -c cfgs/mobile_config.json -w output/mobile.pb -i VOCdevkit/VOC2007/JPEGImages/000009.jpg
#convert-to-uff tensorflow -o output/tiny.uff --input-file output/tiny.pb -O YOLO_output/Reshape
