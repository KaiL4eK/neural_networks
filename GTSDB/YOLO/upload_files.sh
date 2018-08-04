#!/bin/bash

rsync -avzcLP -e "ssh -p 9992"  userquadro@uniq:~/yolo/
