#!/bin/bash

ITER=9000

ln -sf snapshot/snapshot_iter_$ITER.caffemodel weights
ln -sf snapshot/snapshot_iter_$ITER.solverstate state
