#!/bin/bash

rsync -e "ssh -p $UNIQ_PORT" -avzcP data/RF/ alex@uniq:~/NN_data/RF/
