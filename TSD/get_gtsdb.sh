#!/bin/bash

mkdir -p data/GTSDB; cd data/GTSDB

wget -cN https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TestIJCNN2013.zip
wget -cN https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip
wget -cN https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/gt.txt

unzip -fo TrainIJCNN2013.zip
unzip -fo TestIJCNN2013.zip

mv gt.txt TestIJCNN2013Download
