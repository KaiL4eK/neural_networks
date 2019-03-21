import core
from _common.voc import parse_voc_annotation
import os

train_annot_folder = ['/home/alex/Dev/NN_Data/NN_rf_data/rf_tsd_voc/RF17/Annotations_train']
train_image_folder = ['/home/alex/Dev/NN_Data/NN_rf_data/rf_tsd_voc/RF17/Images']

train_annot_folder = ['/home/alex/Dev/NN_Data/NN_rf_data/rf_tsd_voc/Uni/Annotations_train']
train_image_folder = ['/home/alex/Dev/NN_Data/NN_rf_data/rf_tsd_voc/Uni/Images']

train_image_folder = ['/home/alex/catkin_ws/src/AutoNetChallenge/wr8_ai/neural_networks/TSD/rf_tsd_voc/RF19/test/vid1/',
					  '/home/alex/catkin_ws/src/AutoNetChallenge/wr8_ai/neural_networks/TSD/rf_tsd_voc/RF19/test/vid2/',
					  '/home/alex/catkin_ws/src/AutoNetChallenge/wr8_ai/neural_networks/TSD/rf_tsd_voc/RF19/traffic_light/']
train_annot_folder = ['/home/alex/catkin_ws/src/AutoNetChallenge/wr8_ai/neural_networks/TSD/rf_tsd_voc/RF19/Annotations_train/']

train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, None, [])

from shutil import copyfile

output_dir = 'clean_imgs'
if not os.path.isdir(output_dir):
	os.makedirs(output_dir)

for inst in train_ints:
	print(inst['filename'])
	copyfile(inst['filename'], os.path.join(output_dir, os.path.basename(inst['filename'])))
