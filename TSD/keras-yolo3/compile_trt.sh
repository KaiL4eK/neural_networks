#!/bin/bash

if [ ! -f src/numpy.i ]; then
	wget https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i -NP TensorRT/src/
fi

make -C TensorRT
