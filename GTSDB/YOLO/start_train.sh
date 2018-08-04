#!/bin/bash

export PYTHONPATH="$(pwd)/ext_repos/keras-yolo2:$PYTHONPATH"

python ext_repos/keras-yolo2/train.py --conf config-coco-original.json
