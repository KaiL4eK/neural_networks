from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
import random
import itertools

from data import robofest_data_get_samples_preprocessed
from net import *
import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-w', '--weights', action='store', help='Path to weights file')
parser.add_argument('-a', '--augmentation', action='store_true', help='Path to weights file')

args = parser.parse_args()


def print_pretty(str):
	print('-'*30)
	print(str)
	print('-'*30)

def train_and_predict():
	print_pretty('Loading and preprocessing train data...')

	orig_imgs, lane_imgs = robofest_data_get_samples_preprocessed()

	print_pretty('Creating and compiling model...')

	model = get_unet(lr=1e-5)
	batch_size = 10

	if args.weights:
		model.load_weights(args.weights)

	if args.augmentation:
		print_pretty('Setup data generator...')

		data_gen_args = dict( #rotation_range=20,
							  width_shift_range=0.2,
							  height_shift_range=0.2,
							  zoom_range=0.2,
							  horizontal_flip=True,
							  fill_mode='constant',
							  cval=0 )

		image_datagen = ImageDataGenerator(**data_gen_args)
		mask_datagen = ImageDataGenerator(**data_gen_args)

		seed = 1
		image_datagen.fit(imgs_train, augment=True, seed=seed)
		mask_datagen.fit(imgs_mask_train, augment=True, seed=seed)

		print_pretty('Flowing data...')

		# image_generator = image_datagen.flow(imgs_train, batch_size=batch_size, seed=seed, save_to_dir='flow_dir', save_prefix='img_', save_format='png')
		# mask_generator  = mask_datagen.flow(imgs_mask_train, batch_size=batch_size, seed=seed, save_to_dir='flow_dir', save_prefix='mask_', save_format='png')

		image_generator = image_datagen.flow(imgs_train, batch_size=batch_size, seed=seed)
		mask_generator  = mask_datagen.flow(imgs_mask_train, batch_size=batch_size, seed=seed)

		print_pretty('Zipping generators...')

		train_generator = itertools.izip(image_generator, mask_generator)

		print_pretty('Fitting model...')

		model.fit_generator( train_generator, steps_per_epoch=10, epochs=7000, verbose=2,
			callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])
	else:
		print_pretty('Fitting model...')

		model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=7000, verbose=1, shuffle=True, validation_split=0, 
			callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

if __name__ == '__main__':
	train_and_predict()
