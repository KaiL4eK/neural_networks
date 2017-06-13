from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
import random
import itertools

from data import *
from net import *
import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-w', '--weights', action='store', help='Path to weights file')
parser.add_argument('-a', '--augmentation', action='store_true', help='Path to weights file')

args = parser.parse_args()

def preprocess_arrays(imgs, masks):
	imgs_p  = np.ndarray((imgs.shape[0],  nn_img_side, nn_img_side, 3), dtype=np.float32)
	masks_p = np.ndarray((masks.shape[0], nn_out_size, nn_out_size),    dtype=np.float32)

	for i in range(imgs.shape[0]):
		imgs_p[i]  = preprocess_img(imgs[i])
		masks_p[i] = preprocess_mask(masks[i])
		
		# cv2.imshow('1', imgs_p[i])
		# cv2.imshow('2', masks_p[i])
		# if cv2.waitKey(0) == 27:
		# 	exit(1)

	return imgs_p, masks_p[..., np.newaxis]
	# return imgs_p, masks_p


def train_and_predict():
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)

	imgs_train 			= npy_data_load_images()
	imgs_mask_train 	= npy_data_load_masks()
	imgs_train, imgs_mask_train = preprocess_arrays(imgs_train, imgs_mask_train)

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	model = get_unet(lr=1e-4)
	batch_size = 20

	if args.weights:
		model.load_weights(args.weights)

	if args.augmentation:
		print('-'*30)
		print('Setup data generator...')
		print('-'*30)

		data_gen_args = dict( rotation_range=20,
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

		print('-'*30)
		print('Flowing data...')
		print('-'*30)

		# image_generator = image_datagen.flow(imgs_train, batch_size=batch_size, seed=seed, save_to_dir='flow_dir', save_prefix='img_', save_format='png')
		# mask_generator  = mask_datagen.flow(imgs_mask_train, batch_size=batch_size, seed=seed, save_to_dir='flow_dir', save_prefix='mask_', save_format='png')

		image_generator = image_datagen.flow(imgs_train, batch_size=batch_size, seed=seed)
		mask_generator  = mask_datagen.flow(imgs_mask_train, batch_size=batch_size, seed=seed)

		print('-'*30)
		print('Zipping generators...')
		print('-'*30)

		train_generator = itertools.izip(image_generator, mask_generator)

		print('-'*30)
		print('Fitting model...')
		print('-'*30)

		model.fit_generator( train_generator, steps_per_epoch=10, epochs=7000, verbose=2,
			callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])
	else:
		print('-'*30)
		print('Fitting model...')
		print('-'*30)

		model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=7000, verbose=1, shuffle=True, validation_split=0, 
			callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

if __name__ == '__main__':
	train_and_predict()
