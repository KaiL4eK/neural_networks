#!/bin/bash

mkdir -p snapshot
#caffe train --weights weights --solver solver.prototxt 2>&1 | tee learn.log
caffe train --solver solver.prototxt 2>&1 | tee learn.log
