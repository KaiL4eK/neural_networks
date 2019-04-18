#!/bin/bash

tar czvf traffic_trunc_data_lmdb.tar.gz *.prototxt $(readlink weights) $(readlink state) weights state
