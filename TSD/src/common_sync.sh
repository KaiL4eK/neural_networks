#!/bin/bash


if [ -z "$REMOTE_ADDR" ]; then
	echo "REMOTE_ADDR is empty"
	exit
fi

if [ -z "$REMOTE_PORT" ]; then
	echo "REMOTE_PORT is not set"
	exit
fi


OPTIONS="-avzcLPh"

for i in "$@"
do
case $i in
    -f|--full-sync)
		echo "Full sync"
		# Just to sync DB
		rsync -avzcLPe "ssh -p $REMOTE_PORT" ../data $REMOTE_ADDR:$DST/../
    ;;
    -t|--download-output)
		echo "Download results"
		rsync -avzcLPe "ssh -p $REMOTE_PORT" $REMOTE_ADDR:$DST/output/ output/
		exit
    ;;
    -d|--download-weights)
		echo "Download weights"
		rsync -avzcLPe "ssh -p $REMOTE_PORT" $REMOTE_ADDR:$DST/best_chk/ best_chk/
		exit
    ;;
    -u|--upload-weights)
		echo "Upload weights"
		rsync -avzcLPe "ssh -p $REMOTE_PORT" best_chk/ $REMOTE_ADDR:$DST/best_chk/
		exit
    ;;
    -s|--download-src-weights)
		echo "Download source weights"
		rsync -avzcLPe "ssh -p $REMOTE_PORT" $REMOTE_ADDR:$DST/src_weights/ src_weights/
		exit
    ;;
    *)
        echo "Unknown option"
        continue
    ;;
esac
done

rsync -avzcLPe "ssh -p $REMOTE_PORT" \
*.py cfgs utils test src_weights zoo engines src Makefile \
--exclude=.git --exclude=.gitignore --exclude=*.jpg --exclude=test_fld --exclude=__pycache__ \
$REMOTE_ADDR:$DST/
