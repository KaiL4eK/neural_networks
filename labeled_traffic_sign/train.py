from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, Callback, RemoteMonitor
import random

from data import *
import net
import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-b', '--batch_size', default=1, action='store', help='Size of batch to learn')
parser.add_argument('-w', '--weights', action='store',      help='Path to weights file')
# parser.add_argument('-t', '--test',    action='store_true', help='Test model for loss')

args = parser.parse_args()

def load_classification_train_data():
	imgs 			= npy_data_load_images()
	label_list 		= npy_data_load_classes()

	imgs_p   = np.ndarray((imgs.shape[0],  net.nn_img_side, net.nn_img_side, 3), 	dtype=np.float32)
	label_p  = np.ndarray((label_list.shape[0], len(net.glob_label_list)),	dtype=np.float32)

	for i in range(imgs.shape[0]):
		imgs_p[i]   = net.preprocess_img(imgs[i])
		label_p[i] 	= label_list[i]
		# print(label_p[i], label_p[i].shape)

	return imgs_p, label_p


print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

imgs_train, imgs_class_train = load_classification_train_data()

def train_classification():
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	model = net.get_network_model(1e-3)

	if args.weights:
		model.load_weights(args.weights)


	print('-'*30)
	print('Fitting model...')
	print('-'*30)

	input_data  = imgs_train
	output_data = imgs_class_train

	model.fit(input_data, output_data, batch_size=int(args.batch_size), epochs=70000, verbose=1, shuffle=True, validation_split=0.2,
				callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

if __name__ == '__main__':
	train_classification()
