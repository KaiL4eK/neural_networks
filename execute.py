from __future__ import print_function

import os
import cv2
import numpy as np
import time
from keras.models import Model, load_model, save_model

from data import load_train_data, load_test_data
import rects as R
from net import *
import argparse

from skvideo.io import VideoCapture

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('videopath', action='store', help='Path to video file to process')
parser.add_argument('weights', action='store', help='Path to weights file')
parser.add_argument('-f', '--fps', action='store_true', help='Check fps')

args = parser.parse_args()

data_path = 'raw/'

def get_bbox(frame, model):
	imgs_mask = model.predict(np.array([preprocess_img(frame)]))
	mask = (imgs_mask[0] * 255).astype('uint8', copy=False)

	rects = R.get_rects(mask, 100)
	if rects:
		return rects[0]
	else:
		return None

def execute_model():
	cap = VideoCapture(args.videopath)
	if cap is None:
		exit(1)

	

	ret, frame = cap.read()
	f_height, f_width, f_cahnnels = frame.shape
	scale_x = float(f_width)/nn_img_side
	scale_y = float(f_height)/nn_img_side

	# model = load_model('test_model.h5', custom_objects={'iou_loss':iou_loss})
	model = get_unet()
	model.load_weights(args.weights)

	if args.fps:
		num_frames = 120
		start = time.time()
		for i in xrange(0, num_frames) :
			ret, frame = cap.read()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			bbox = get_bbox(frame, model)

			if bbox:
				# print(bbox)
				bbox = [int(i) for i in np.multiply(bbox, [scale_x, scale_y, scale_x, scale_y])]
				# print(bbox)
				R.draw_rects(frame, [bbox])
			
			cv2.imshow('frame',frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		end = time.time()
		seconds = end - start
		print('Time taken : {0} seconds'.format(seconds))
		fps  = num_frames / seconds;
		print('Estimated frames per second : {0}'.format(fps))

	else:	
		while(cap.isOpened()):
			ret, frame = cap.read()

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			bbox = get_bbox(frame, model)

			if bbox:
				# print(bbox)
				bbox = [int(i) for i in np.multiply(bbox, [scale_x, scale_y, scale_x, scale_y])]
				# print(bbox)
				R.draw_rects(frame, [bbox])
			
			cv2.imshow('frame',frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	cap.release()
	cv2.destroyAllWindows()

	# for image_name in os.listdir(data_path):
	# 	img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_COLOR)
	# 	cv2.imshow('1', img)
	# 	height, width, channels = img.shape
		
	# 	img = preprocess_image(img)

	# 	imgs_mask = model.predict(np.array([img]))
	# 	img_mask = cv2.resize(imgs_mask[0], (width, height), interpolation = cv2.INTER_NEAREST)
	# 	cv2.imshow('2', img_mask)

	# 	if cv2.waitKey(0) == 27:
	# 		exit(1)


if __name__ == '__main__':
	execute_model()
