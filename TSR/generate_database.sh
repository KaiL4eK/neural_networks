#!/bin/bash

python scripts/voc_generate_tsr.py -i ../TSD/rf_tsd_voc/RF17/Images -a ../TSD/rf_tsd_voc/RF17/Annotations_train -o signs_ds
python scripts/voc_generate_tsr.py -i ../TSD/rf_tsd_voc/Uni/Images -a ../TSD/rf_tsd_voc/Uni/Annotations_train -o signs_ds


