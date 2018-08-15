#!/bin/bash

python export_2_pb.py -c cfgs/tiny_config_cont.json -w chk/tiny_yolov2_bestMap.h5 -o tiny
#convert-to-uff tensorflow -o output/tiny.uff --input-file output/tiny.pb -O YOLO_output/Reshape
