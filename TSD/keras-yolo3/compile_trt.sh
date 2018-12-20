#!/bin/bash

TRT_SRC_BASE=TensorRT/src

if [ ! -f $TRT_SRC_BASE/numpy.i ]; then
	wget https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i -NP $TRT_SRC_BASE
fi

make -C TensorRT
