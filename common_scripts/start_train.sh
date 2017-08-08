#!/bin/bash

python train.py -w weights_best.h5 -ab 30 -l 1e-4 2>&1 | tee learn.log
