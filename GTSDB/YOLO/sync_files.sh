#!/bin/bash

for i in "$@"
do
case $i in
    -f|--full-sync)
		echo "Full sync"
		# Just to sync DB
		rsync -avzcLP -e "ssh -p 9992" VOCdevkit userquadro@uniq:~/yolo/
    ;;
    -d|--download-weights)
		echo "Download weights"
		# Just to sync DB
		rsync -avzcLP -e "ssh -p 9992" userquadro@uniq:~/yolo/chk .
		exit
    ;;
    -u|--upload-weights)
		echo "Upload weights"
		# Just to sync DB
		rsync -avzcLP -e "ssh -p 9992" chk/ userquadro@uniq:~/yolo/chk/
		exit
    ;;
    *)
          # unknown option
    ;;
esac
done

rsync -avzcLP -e "ssh -p 9992" \
 \
ext_repos *.py cfgs \
--exclude=.git --exclude=.gitignore --exclude=*.jpg --exclude=test_fld --exclude=__pycache__ \
userquadro@uniq:~/yolo/

