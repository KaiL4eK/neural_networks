#!/bin/bash

rsync -e "ssh -p $UNIQ_PORT" -avzcP data/RF/TSD/RF17/ alex@uniq:~/NN_data/RF/TSD/RF17/
rsync -e "ssh -p $UNIQ_PORT" -avzcP data/RF/TSD/RF19/ alex@uniq:~/NN_data/RF/TSD/RF19/
