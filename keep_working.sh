#!/bin/bash

while :
do
	ps cax | grep start_train.sh > /dev/null

	if [ $? -eq 0 ]; then
  		echo "Process is running."
	else
  		echo "Process is not running." >> checking.log
  		./start_train.sh
	fi
	sleep 5
done
