#!/bin/bash

#python train.py -w last_weights_best.h5 2>&1 | tee learn.log
python train.py -w weights_best.h5 -b 30 2>&1 | tee learn.log
#python train.py 2>&1 | tee learn.log
