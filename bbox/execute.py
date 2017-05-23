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
parser.add_argument('filepath', action='store', help='Path to video file to process')
parser.add_argument('weights', action='store', help='Path to weights file')
parser.add_argument('-f', '--fps', action='store_true', help='Check fps')
parser.add_argument('-p', '--pic', action='store_true', help='Process picture')

args = parser.parse_args()

data_path = 'raw/'

try:
    xrange
except NameError:
    xrange = range

def get_bbox(frame, model):
	imgs_mask = model.predict(np.array([preprocess_img(frame)]))
	f_height, f_width, f_cahnnels = frame.shape
	result = imgs_mask[0] * [f_width, f_height, f_width, f_height]
	return result.astype(int)

def execute_model():

	model = regression_model()

	if args.pic:
		
		frame = cv2.imread(args.filepath)
		if frame is None:
			print('Failed to open file')
			exit(1)
		
		print_summary(model)
		model.load_weights(args.weights)

		bbox = get_bbox(frame, model)

		print(bbox)
		R.draw_rects(frame, [bbox])
		
		cv2.imshow('frame',frame)
		cv2.waitKey(0)
		exit(1)
	else:

		cap = cv2.VideoCapture(args.filepath)
		if cap is None or not cap.isOpened():
			print('Failed to open file')
			exit(1)

		ret, frame = cap.read()
		if frame is None:
			print('Failed to read frame')
			exit(1)

		print_summary(model)
		model.load_weights(args.weights)

		if args.fps:
			bbox_obtain_time = 0
			num_frames = 120
			start = time.time()
			for i in xrange(0, num_frames) :
				ret, frame = cap.read()
				if frame is None:
					exit(1)

				start_bbox = time.time()
				bbox = get_bbox(frame, model)
				bbox_obtain_time += (time.time() - start_bbox)

				print(bbox)
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

				bbox = get_bbox(frame, model)
						
				print(bbox)
				R.draw_rects(frame, [bbox])
				
				cv2.imshow('frame',frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

		cap.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	execute_model()
