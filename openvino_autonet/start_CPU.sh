#!/bin/bash

BASE_NET=MbN2_416x416_t1

./build/app -d CPU -i test.jpg \
    -r ../TSD/src/_gen/ir_models/"$BASE_NET"_FP16.xml \
    -c ../TSD/src/_gen/ir_models/$BASE_NET.json
