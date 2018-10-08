#!/bin/bash

rm -rf test/detected/*  && python predict.py -c cfgs/tiny_config_cont.json -w chk/tiny_yolov2_bestMap.h5 -i test
