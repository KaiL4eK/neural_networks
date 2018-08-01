#!/bin/bash

rsync -avzLP -e "ssh -p 9992" train.py data.py model.py *.npy userquadro@uniq:~/keras_NN/
