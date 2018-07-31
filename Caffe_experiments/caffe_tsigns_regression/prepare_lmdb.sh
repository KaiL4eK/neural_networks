#!/bin/bash

rm -rf data; mkdir data &&\
python create_lmdb.py 

# cd data &&\
# compute_image_mean input_train_lmdb input_train_mean.binaryproto &&\
# compute_image_mean input_verify_lmdb input_verify_mean.binaryproto
