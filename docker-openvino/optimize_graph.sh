#!/bin/bash

MO_PATH=/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py
GRAPH_PATH=$1
INPUT_SHAPE="[2,416,416,3]"
OUTPUT_DIR="openvino_models"
OUTPUT_MODEL="sample"

OPTIM_DIR="for_optimization"

mkdir -p $OPTIM_DIR; cp $GRAPH_PATH $OPTIM_DIR;
GRAPH_FNAME=`basename $GRAPH_PATH`

docker run --rm -it --privileged \
    -v /dev:/dev -v `pwd`:/home/developer/ws \
    docker-openvino \
    bash -c \
    "python3 $MO_PATH --input_model $OPTIM_DIR/$GRAPH_FNAME --input_shape $INPUT_SHAPE --output_dir $OUTPUT_DIR --data_type FP16"

# --model_name '$OUTPUT_MODEL'_fp32
#  --log_level=DEBUG
