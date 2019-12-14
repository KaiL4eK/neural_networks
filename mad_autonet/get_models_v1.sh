#!/usr/bin/env bash

HOST="194.85.169.205:8000"

mkdir -p models; cd models

FP16_MODEL="MbN2_416x416_t1_FP16"
FP32_MODEL="MbN2_416x416_t1_FP32"

wget -N "http://$HOST/test_data/RF_model_YOLO/MbN2_416x416_t1.json"

wget -N "http://$HOST/test_data/RF_model_YOLO/$FP16_MODEL.bin"
wget -N "http://$HOST/test_data/RF_model_YOLO/$FP16_MODEL.mapping"
wget -N "http://$HOST/test_data/RF_model_YOLO/$FP16_MODEL.xml"

wget -N "http://$HOST/test_data/RF_model_YOLO/$FP32_MODEL.bin"
wget -N "http://$HOST/test_data/RF_model_YOLO/$FP32_MODEL.mapping"
wget -N "http://$HOST/test_data/RF_model_YOLO/$FP32_MODEL.xml"
