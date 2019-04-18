#!/bin/bash

tar czvf traffic_deconv_data.tar.gz data *.prototxt start_caffe.sh restore_caffe.sh $(readlink weights) $(readlink state) weights state make_links.sh make_data_archive.sh make_trunc_data_archive.sh
