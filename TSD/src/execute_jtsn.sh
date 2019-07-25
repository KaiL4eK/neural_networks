#!/bin/bash

./sync_files_jtsn.sh

ssh $JETSON_ADDR 'cd yolo3; python3 predict_show.py -c cfgs/signs_tiny_v3.json  -w best_chk/sign_tiny_v3_best_mAP0.618.h5 -i ../data/GTSDB_voc/Images'
# ssh $JETSON_ADDR 'cd yolo3; python3 trt_inf.py -i ../data/GTSDB_voc/Images -e engines/trtEngine_Tiny.trt'
