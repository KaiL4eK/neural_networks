#!/bin/bash

for i in "$@"
do
case $i in
    -f|--full-sync)
		echo "Full sync"
		# Just to sync DB
		rsync -avzcLP -e "ssh -p 9992" VOCdevkit userquadro@uniq:~/yolo/
    ;;
    -c|--config-download)
		echo "Full sync"
		# Just to sync DB
		rsync -avzcLP -e "ssh -p 9992" userquadro@uniq:~/yolo/config.json .
    ;;
    *)
          # unknown option
    ;;
esac
done

rsync -avzcLP -e "ssh -p 9992" \
mobilenet_backend.h5 *.json ext_repos *.py *.sh \
--exclude=.git --exclude=test_fld --exclude=__pycache__ \
userquadro@uniq:~/yolo/

