#!/bin/bash

python export_2_pb.py -c cfgs/tiny_config.json -w chk/tiny_yolov2_bestMap.h5 -o output/tiny.pb 		&& \
convert-to-uff tensorflow -o output/tiny.uff --input-file output/tiny.pb -O YOLO_output/Reshape




# python export_2_pb.py -c cfgs/mobile_config.json -w chk/mobile_bestMap.h5 -o mobile && \
# convert-to-uff tensorflow -l --input-file output/mobile.pb && \
# python tensorrt_inf.py -c cfgs/mobile_config.json -w output/mobile.pb -i ../../data_root/VOCdevkit_test/VOC2007/JPEGImages/000001.jpg
# convert-to-uff tensorflow -o output/mobile.uff --input-file output/mobile.pb -O YOLO_output/Reshape
