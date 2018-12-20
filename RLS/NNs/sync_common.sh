#!/bin/bash

if [ -z "$REMOTE_ADDR" ]; then
	echo "REMOTE_ADDR is empty"
	exit
fi

if [ -z "$REMOTE_PORT" ]; then
	echo "REMOTE_PORT is not set"
	exit
fi

OPTIONS="-avzcLPhe"

for i in "$@"
do
case $i in
    -f|--full-sync)
		echo "Full sync"
		# Just to sync DB
		rsync "$OPTIONS ssh -p $REMOTE_PORT" ../data $REMOTE_ADDR:$DST/../
    ;;
    -t|--download-output)
		echo "Download results"
		rsync "$OPTIONS ssh -p $REMOTE_PORT" $REMOTE_ADDR:$DST/output/ output/
		exit
    ;;
    -d|--download-weights)
		echo "Download weights"
		rsync "$OPTIONS ssh -p $REMOTE_PORT" $REMOTE_ADDR:$DST/best_chk/ best_chk/
		exit
    ;;
    -u|--upload-weights)
		echo "Upload weights"
		rsync "$OPTIONS ssh -p $REMOTE_PORT" best_chk/ $REMOTE_ADDR:$DST/best_chk/
		exit
    ;;
    -s|--download-src-weights)
		echo "Download source weights"
		rsync "$OPTIONS ssh -p $REMOTE_PORT" $REMOTE_ADDR:$DST/src_weights/ src_weights/
		exit
    ;;
    *)
        echo "Unknown option"
        continue
    ;;
esac
done

rsync "$OPTIONS ssh -p $REMOTE_PORT" * \
--exclude=.git --exclude=.gitignore --exclude=*.jpg --exclude=__pycache__ --exclude=best_chk --exclude=output --exclude=logs \
$REMOTE_ADDR:$DST/