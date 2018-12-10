from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback

from keras.utils.layer_utils import print_summary
from keras.utils.vis_utils import plot_model

import random

from data import robofest_data_get_samples_preprocessed
from net import create_model
import argparse

from utils import print_pretty, init_gpu_session

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-w', '--weights', action='store', help='Path to weights file')
parser.add_argument('-a', '--augmentation', action='store_true', help='Path to weights file')
parser.add_argument('-s', '--structure', action='store_true', help='Show only structure of network')

args = parser.parse_args()

init_gpu_session(0.7)

def train_and_predict():
	
	pretrained_weights_path = args.weights
	structure_mode = args.structure


	print_pretty('Creating and compiling model...')

	train_model, infer_model = create_model(input_shape=(320, 640, 3), lr=1e-4)
	

	if structure_mode:
		print_summary(train_model)
		# plot_model(model, show_shapes=True)

		return

	if pretrained_weights_path:
		train_model.load_weights(args.weights)


	print_pretty('Loading and preprocessing train data...')

	orig_imgs, mask_imgs = robofest_data_get_samples_preprocessed()


	print_pretty('Setup data generator...')

	imgs_train 		= orig_imgs
	imgs_mask_train = mask_imgs

	data_gen_args = dict( rotation_range=5,
						  width_shift_range=0.1,
						  height_shift_range=0.1,
						  zoom_range=0.1,
						  horizontal_flip=True,
						  fill_mode='constant',
						  cval=0 )

	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)

	seed = 1
	batch_size = 4
	image_datagen.fit(imgs_train, augment=True, seed=seed)
	mask_datagen.fit(imgs_mask_train, augment=True, seed=seed)


	print_pretty('Flowing data...')

	# image_generator = image_datagen.flow(imgs_train, batch_size=batch_size, seed=seed, save_to_dir='flow_dir', save_prefix='img_', save_format='png')
	# mask_generator  = mask_datagen.flow(imgs_mask_train, batch_size=batch_size, seed=seed, save_to_dir='flow_dir', save_prefix='mask_', save_format='png')

	image_generator = image_datagen.flow(imgs_train, batch_size=batch_size, seed=seed)
	mask_generator  = mask_datagen.flow(imgs_mask_train, batch_size=batch_size, seed=seed)


	print_pretty('Zipping generators...')

	train_generator = zip(image_generator, mask_generator)

	print_pretty('Fitting model...')

	train_model.fit_generator( train_generator, steps_per_epoch=10, epochs=100, verbose=1,
		callbacks=[ModelCheckpoint('chk/weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=False, verbose=1)])


	# 	model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=7000, verbose=1, shuffle=True, validation_split=0, 
	# 		callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

if __name__ == '__main__':
	train_and_predict()
