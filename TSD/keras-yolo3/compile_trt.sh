#!/bin/bash

if [ ! -f src/numpy.i ]; then
	wget https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i -P src/
fi


