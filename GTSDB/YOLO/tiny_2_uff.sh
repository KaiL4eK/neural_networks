#!/bin/bash

python export_2_pb.py -c cfgs/tiny_config.json -w chk/tiny_yolov2_bestMap.h5 -o tiny && \
convert-to-uff tensorflow -l --input-file output/tiny.pb && \
python tensorrt_inf.py -c cfgs/tiny_config.json -w output/tiny.pb -i VOCdevkit/VOC2007/JPEGImages/000009.jpg
#convert-to-uff tensorflow -o output/tiny.uff --input-file output/tiny.pb -O YOLO_output/Reshape
