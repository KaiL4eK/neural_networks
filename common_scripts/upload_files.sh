#!/bin/bash

rsync -avzL train.py data.py net.py start_train.sh *.npy usergpu@uni:~/keras_classification_NN