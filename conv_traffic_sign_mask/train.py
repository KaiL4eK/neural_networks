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
parser.add_argument('-b', '--batch_size', default=1, action='store', help='Size of batch to learn')
parser.add_argument('-a', '--augmentation', action='store_true', help='Path to weights file')

args = parser.parse_args()

def preprocess_arrays(imgs, masks):
	imgs_p  	 = np.ndarray((imgs.shape[0],  nn_img_h, nn_img_w, 3), 			  dtype=np.float32)
	# masks_p 	 = np.ndarray((masks.shape[0], nn_out_h, nn_out_w),    			  dtype=np.float32)
	grid_masks_p = np.ndarray((masks.shape[0], nn_grid_y_count, nn_grid_x_count), dtype=np.float32)

	for i in range(imgs.shape[0]):
		for y in range(nn_grid_y_count):
			for x in range(nn_grid_x_count):
				x_px, y_px = (x*nn_next_size, y*nn_next_size)
				# print('Cell %d to %d' % (x_px, x_px + nn_next_size))
				img_part = masks[i][y_px : y_px + nn_next_size, x_px : x_px + nn_next_size]

				if np.sum(img_part) > 0:
					grid_masks_p[i][y, x] = 1.
				else:
					grid_masks_p[i][y, x] = 0

		# show_mask_g = cv2.resize(grid_masks_p[i], (600, 300), interpolation = cv2.INTER_NEAREST)
		# show_mask_r = cv2.resize(masks[i], 	   (600, 300), interpolation = cv2.INTER_NEAREST)

		# show_mask_g = cv2.cvtColor(show_mask_g, cv2.COLOR_GRAY2BGR)
		# show_mask_g[np.where((show_mask_g == (0, 0, 0)).all(axis=2))] = (0,0,255)
		# show_mask_g[np.where((show_mask_r != 0))] = (255,0,0)

		# cv2.imshow('1', show_mask_g)
		# # cv2.imshow('2', imgs[i])
		# if cv2.waitKey(0) == 27:
		# 	exit(1)

	for i in range(imgs.shape[0]):
		imgs_p[i]  = preprocess_img(imgs[i])
		# masks_p[i] = preprocess_mask(masks[i])

	p = np.random.permutation(len(imgs_p))
	imgs_p  		= imgs_p[p]
	# masks_p 		= masks_p[p]
	grid_masks_p	= grid_masks_p[p]


	# if 1:
	# 	for i in range(imgs.shape[0]):
	# 		cv2.imshow('1', imgs_p[i])
	# 		cv2.imshow('2', masks_p[i])
	# 		if cv2.waitKey(0) == 27:
	# 			exit(1)

	# fl = masks_p.flatten()
	# lg = np.log(np.clip(fl, 1e-9, 1))
	# print(np.mean(np.multiply(lg, fl)))
	# cv2.imshow('1', masks[0])
	# cv2.waitKey(3000)

	return imgs_p, grid_masks_p[..., np.newaxis]


def train_and_predict():
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)

	imgs_train 			= npy_data_load_images()
	imgs_mask_train 	= npy_data_load_masks()
	imgs_train, imgs_mask_train = preprocess_arrays(imgs_train, imgs_mask_train)

	if len( np.unique(imgs_mask_train) ) > 2:
		print('Preprocessing created mask with more than two binary values')
		exit(1)

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	model = get_unet(lr=1e-4)
	batch_size = int(args.batch_size)

	print('Batch size is set to %d' % batch_size)

	if args.weights:
		model.load_weights(args.weights)

	if args.augmentation:
		print('-'*30)
		print('Setup data generator...')
		print('-'*30)

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

		model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=7000, verbose=1, shuffle=True, validation_split=0.2, 
			callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

if __name__ == '__main__':
	train_and_predict()
