#!/bin/bash

mvNCProfile output/laneseg.pb -in input_img -on flatten_1/Reshape -s 12
# mvNCCompile output/laneseg.pb -in input_img -on conv2d_18/Relu -o output/laneseg.graph -s 12
