from __future__ import print_function

import os
import cv2
import numpy as np
import time
from keras.models import Model, load_model

from data import load_train_data, load_test_data

from net import *

from skvideo.io import VideoCapture


data_path = 'raw/'

def preprocess_image(img):
	img = cv2.resize(img, (img_side_size, img_side_size), interpolation = cv2.INTER_LINEAR)

	img = img.astype('float32')
	img -= mean
	img /= 255.

	return img

def execute_model():

	cap = VideoCapture('vid.webm')

	num_frames = 120
	# cap = cv2.VideoCapture('vid1.avi')
	if cap is None:
		exit(1)

	# exit(1)

	model = get_unet()
	model.load_weights('best_model.h5')

	start = time.time()
	while(cap.isOpened()):
		ret, frame = cap.read()

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		cv2.imshow('frame',frame)
		img = preprocess_image(frame)
		imgs_mask = model.predict(np.array([img]))
		cv2.imshow('2', imgs_mask[0])

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	end = time.time()
	seconds = end - start
	print('Time taken : {0} seconds'.format(seconds))
	fps  = num_frames / seconds;
	print('Estimated frames per second : {0}'.format(fps))

	cap.release()
	cv2.destroyAllWindows()
	exit(1)


	for image_name in os.listdir(data_path):
		img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_COLOR)
		cv2.imshow('1', img)
		height, width, channels = img.shape
		
		img = preprocess_image(img)

		imgs_mask = model.predict(np.array([img]))
		img_mask = cv2.resize(imgs_mask[0], (width, height), interpolation = cv2.INTER_NEAREST)
		cv2.imshow('2', img_mask)

		if cv2.waitKey(0) == 27:
			exit(1)


if __name__ == '__main__':
	execute_model()
