#!/bin/bash

BASE_NET=MbN2_416x416_t1

./build/app -d CPU -i test.jpg \
    -r models/"$BASE_NET"_FP16.xml \
    -c models/$BASE_NET.json
