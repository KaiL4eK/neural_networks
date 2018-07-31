#!/bin/bash

mkdir -p snapshot
caffe train --snapshot state --solver solver.prototxt 2>&1 | tee learn.log
