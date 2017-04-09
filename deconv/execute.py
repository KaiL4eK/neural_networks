from __future__ import print_function

import os
import cv2
import numpy as np
import time
from keras.models import Model, load_model, save_model

from data import load_train_data
import rects as R
from net import *
import argparse

# from skvideo.io import VideoCapture

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('videopath', action='store', help='Path to video file to process')
parser.add_argument('weights', action='store', help='Path to weights file')
parser.add_argument('-f', '--fps', action='store_true', help='Check fps')

args = parser.parse_args()

data_path = 'raw/'

try:
    xrange
except NameError:
    xrange = range

def get_bbox(frame, model):
	imgs_mask = model.predict(np.array([preprocess_img(frame)]))
	mask = (imgs_mask[0] * 255).astype('uint8', copy=False)

	rects = R.get_rects(mask, 100)
	if rects:
		return rects[0]
	else:
		return None

def execute_model():
	cap = cv2.VideoCapture(args.videopath)
	if cap is None or not cap.isOpened():
		print('Failed to open file')
		exit(1)

	ret, frame = cap.read()
	if frame is None:
		print('Failed to read frame')
		exit(1)

	f_height, f_width, f_cahnnels = frame.shape
	scale_x = float(f_width)/nn_out_size
	scale_y = float(f_height)/nn_out_size

	# model = load_model('test_model.h5', custom_objects={'iou_loss':iou_loss})
	model = get_unet()
	print_summary(model)
	# model.load_weights(args.weights)

	if args.fps:
		bbox_obtain_time = 0
		num_frames = 120
		start = time.time()
		for i in xrange(0, num_frames) :
			ret, frame = cap.read()
			if frame is None:
				exit(1)

			# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			start_bbox = time.time()
			bbox = get_bbox(frame, model)
			bbox_obtain_time += (time.time() - start_bbox)

			if bbox:
				# print(bbox)
				bbox = [int(i) for i in np.multiply(bbox, [scale_x, scale_y, scale_x, scale_y])]
				# print(bbox)
				R.draw_rects(frame, [bbox])
			
			cv2.imshow('frame',frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		seconds = time.time() - start
		print('Mean time per frame : {0} ms'.format(seconds / num_frames * 1000))
		print('Estimated frames per second : {0}'.format(num_frames / seconds))
		print('Bbox obtain mean time : {0} ms'.format(bbox_obtain_time / num_frames * 1000))

	else:	
		while(cap.isOpened()):
			ret, frame = cap.read()
			if frame is None:
				exit(1)

			# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

if __name__ == '__main__':
	execute_model()
