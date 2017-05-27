from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
import random

from data import *
from net import *
import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-w', '--weights', action='store', help='Path to weights file')

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

	model = get_unet()

	if args.weights:
		model.load_weights(args.weights)

	print('-'*30)
	print('Fitting model...')
	print('-'*30)

	model.fit(imgs_train, imgs_mask_train, batch_size=10, epochs=7000, verbose=1, shuffle=True, validation_split=0.05, 
				callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

if __name__ == '__main__':
	train_and_predict()
