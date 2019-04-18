#!/bin/bash

tar czvf traffic_deconv_weights_structs.tar.gz *.prototxt $(readlink weights) $(readlink state) weights state
