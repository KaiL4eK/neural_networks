#!/bin/bash

FNAME="RobofestDetection_$(date +%Y%m%d).tar.gz"

ssh -p 9992 alex@uniq "cd ~/NN_data/RF; tar czvf $FNAME TSD"
