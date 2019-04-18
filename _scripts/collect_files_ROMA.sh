#!/bin/bash

# find . -regex ".*\.\(jpg\|gif\|png\|jpeg\)" -print0 | xargs -0 -I {} cp {} ./ROMA/
origins=$(find . -regex ".*\.\(jpg\|gif\|png\|jpeg\)")
references=$(find . -regex ".*\.\(pgm\)")

# echo ${#origins[@]}
# echo ${#references[@]}

# echo $origins
# echo $references

DST_DIR="ROMA"

mkdir -p $DST_DIR

for origin in $origins; do
	echo $origin

	new_fname="$(dirname $origin)/R$(basename $origin)"
	ref_name="${new_fname%.*}.pgm"
	new_ref_name=$(basename $origin)
	new_ref_name="${new_ref_name%.*}.ref"
	echo $ref_name
	echo $new_ref_name

	if [ ! -f $ref_name ]; then
		echo "Fail!"
		exit 1

	fi

	cp $origin $DST_DIR/
	cp $ref_name $DST_DIR/$new_ref_name

	echo "-----------"
done
