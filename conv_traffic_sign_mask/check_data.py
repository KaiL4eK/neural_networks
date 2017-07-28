from __future__ import print_function

import os
import cv2
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import load_train_data, load_test_data

def check_data():

	imgs_test, imgs_id_test = load_test_data()
	imgs_mask_test = np.load('imgs_mask_test.npy')
	print('-' * 30)
	print('Saving predicted masks to files...')
	print('-' * 30)
	pred_dir = 'preds/'
	if not os.path.exists(pred_dir):
		os.mkdir(pred_dir)

	print(imgs_mask_test.shape)
	print(imgs_id_test.shape)

	for image, image_id in zip(imgs_mask_test, imgs_id_test):
		
		# print(image.shape)
		image = (image[:, :, 0] * 255.).astype(np.uint8)

		# print(image.shape)
		#cv2.imshow('1', image)
		#if cv2.waitKey(0) == 27:
		#    exit( 1 )

		imsave('preds/{}_pred.png'.format(image_id), image)
		# cv2.imwrite(pred_dir + image_id + '_pred.png', image)

if __name__ == '__main__':
	check_data()