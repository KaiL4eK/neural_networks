#!/bin/bash

tar czvf traffic_data_lmdb.tar.gz data *.prototxt start_caffe.sh restore_caffe.sh $(readlink weights) $(readlink state) weights state
