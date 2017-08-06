from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, Callback, RemoteMonitor
from keras.preprocessing.image import ImageDataGenerator
import random

from data import *
import net
import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-b', '--batch_size', 	 default=1, 	action='store', 	 help='Size of batch to learn')
parser.add_argument('-w', '--weights', 						action='store', 	 help='Path to weights file')
parser.add_argument('-l', '--learning_rate', default=1e-4,	action='store', 	 help='Learning rate value')
parser.add_argument('-a', '--augmentation', 				action='store_true', help='Use augmentation')

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

	print('Learning rate: {} / Batch size: {}'.format(float(args.learning_rate), int(args.batch_size)))

	model = net.get_network_model(float(args.learning_rate))

	if args.weights:
		model.load_weights(args.weights)

	input_data  = imgs_train
	output_data = imgs_class_train

	save_callback = ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)
	epochs = 7000

	if args.augmentation:
		print('-'*30)
		print('Fitting model...')
		print('-'*30)

		datagen = ImageDataGenerator(
		    rotation_range=5,
		    width_shift_range=0.05,
		    height_shift_range=0.05,
		    horizontal_flip=True)

		data_generator = datagen.flow(input_data, output_data, batch_size = int(args.batch_size))

		# fits the model on batches with real-time data augmentation:
		model.fit_generator(data_generator, steps_per_epoch=len(input_data) / int(args.batch_size), epochs=epochs, 
							validation_data=(input_data, output_data),
							callbacks=[save_callback])
	else:
		print('-'*30)
		print('Fitting model...')
		print('-'*30)

		model.fit(  input_data, output_data, batch_size=int(args.batch_size), epochs=epochs, 
					verbose=1, shuffle=True, validation_split=0.2,
					callbacks=[save_callback])

if __name__ == '__main__':
	train_classification()
