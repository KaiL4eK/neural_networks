#!/bin/bash

rsync -avzL train.py data.py net.py start_train.sh weights_best.h5 *.npy usergpu@uni:~/keras_NN/
