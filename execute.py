from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model

from data import load_train_data, load_test_data

from net import *

data_path = 'raw/'

def preprocess_image(img):
	img = cv2.resize(img, (img_side_size, img_side_size), interpolation = cv2.INTER_LINEAR)

	img = img.astype('float32')
	img -= mean
	img /= 255.

	return img

def execute_model():
	model = get_unet()
	model.load_weights('last_weights.h5')

	for image_name in os.listdir(data_path):
		img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_COLOR)
		cv2.imshow('1', img)
		height, width, channels = img.shape
		
		img = preprocess_image(img)

		imgs_mask = model.predict(np.array([img]), verbose=1)
		img_mask = (imgs_mask[0, :, :, 0] * 255.).astype(np.uint8)

		img_mask = cv2.resize(imgs_mask[0], (width, height), interpolation = cv2.INTER_NEAREST)
		cv2.imshow('2', img_mask)

		if cv2.waitKey(0) == 27:
			exit(1)


if __name__ == '__main__':
	execute_model()
