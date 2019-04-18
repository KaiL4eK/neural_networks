#!/bin/bash

mkdir -p snapshot
python solve.py 2>&1 | tee learn.log
