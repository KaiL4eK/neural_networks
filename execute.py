from __future__ import print_function

import os
import cv2
import numpy as np
import time
from keras.models import Model, load_model

from data import load_train_data, load_test_data
import rects as R
from net import *
import argparse

from skvideo.io import VideoCapture

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('filepath', action='store', help='Path to video file to process')

args = parser.parse_args()

data_path = 'raw/'

def execute_model():

	cap = VideoCapture(args.filepath)

	num_frames = 120
	# cap = cv2.VideoCapture('vid1.avi')
	if cap is None:
		exit(1)

	# exit(1)

	ret, frame = cap.read()
	f_height, f_width, f_cahnnels = frame.shape
	scale_x = float(f_width)/img_side_size
	scale_y = float(f_height)/img_side_size

	model = get_unet()
	model.load_weights('best_model.h5')

	start = time.time()
	# while(cap.isOpened()):
	for i in xrange(0, num_frames) :
		ret, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		imgs_mask = model.predict(np.array([preprocess_img(frame)]))
		mask = (imgs_mask[0] * 255).astype('uint8', copy=False)

		rects = R.get_rects(mask, 100)
		if rects:
			rect = rects[0]
			# print(rect)

			rect = [int(i) for i in np.multiply(rect, [scale_x, scale_y, scale_x, scale_y])]

			# print(rect)

			R.draw_rects(frame, [rect])
		
		cv2.imshow('frame',frame)
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
