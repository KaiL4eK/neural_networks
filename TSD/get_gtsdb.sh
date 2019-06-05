#!/bin/bash

mkdir -p data/GTSDB; cd data/GTSDB

wget -cN https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TestIJCNN2013.zip
wget -cN https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip
wget -cN https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/gt.txt

unzip -oq TrainIJCNN2013.zip
unzip -oq TestIJCNN2013.zip

mv gt.txt TestIJCNN2013Download

cd ../../

python parser_gtsdb_2_voc.py -i data/GTSDB/TrainIJCNN2013 -o data/GTSDB_voc_train
python parser_gtsdb_2_voc.py -i data/GTSDB/TestIJCNN2013Download -o data/GTSDB_voc_test
