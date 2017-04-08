from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
import random

from data import load_train_data, load_test_data, data_path
from net import *

def preprocess_arrays(imgs, masks):
	imgs_p  = np.ndarray((imgs.shape[0],  img_side_size, img_side_size, 3), dtype=np.float32)
	masks_p = np.ndarray((masks.shape[0], img_side_size, img_side_size),    dtype=np.float32)
	for i in range(imgs.shape[0]):
		imgs_p[i]  = preprocess_img(imgs[i])
		masks_p[i] = preprocess_mask(masks[i])
		
		# cv2.imshow('1', imgs_p[i])
		# cv2.imshow('2', masks_p[i])
		# if cv2.waitKey(0) == 27:
		# 	exit(1)

	return imgs_p, masks_p[..., np.newaxis]


def evaluate_loss():
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, imgs_mask_train = load_train_data()
	imgs_train, imgs_mask_train = preprocess_arrays(imgs_train, imgs_mask_train)

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	model = get_unet()
	model.load_weights('best_model.h5')

	print('-'*30)
	print('Fitting model...')
	print('-'*30)

	loss = model.evaluate(imgs_train, imgs_mask_train, batch_size=32)
	print(loss[0])

if __name__ == '__main__':
	evaluate_loss()
