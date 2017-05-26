#!/bin/bash

tar czvf keras_deconv_data.tar.gz train.py data.py net.py *.npy
scp keras_deconv_data.tar.gz user8203@uni:~/keras_deconv_tsigns
