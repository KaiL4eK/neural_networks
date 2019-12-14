#!/bin/bash

#OPENVINO_PATH=$HOME/intel/openvino
#export MO_PATH=$OPENVINO_PATH/deployment_tools/model_optimizer/mo_tf.py

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$OPENVINO_PATH/deployment_tools/inference_engine/external/mkltiny_lnx/lib"

. $OPENVINO_PATH/bin/setupvars.sh

