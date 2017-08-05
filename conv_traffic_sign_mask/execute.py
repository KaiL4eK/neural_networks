from __future__ import print_function

import os
import cv2
import numpy as np
import time
from keras.models import Model, load_model, save_model

from data import *
from net import *
import argparse

# from skvideo.io import VideoCapture

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('weights', action='store', help='Path to weights file')
parser.add_argument('filepath', action='store', help='Path to video file to process')
parser.add_argument('-f', '--fps', action='store_true', help='Check fps')
parser.add_argument('-p', '--pic', action='store_true', help='Process picture')

args = parser.parse_args()

data_path = 'raw/'
new_shape = (600, 600)

try:
    xrange
except NameError:
    xrange = range

def get_mask(frame, model):
	img_mask = model.predict(np.array([preprocess_img(frame)]))
	mask = (img_mask[0] * 255).astype('uint8', copy=False)
	

	return mask

def mark_image(frame, model):
	mask = get_mask(frame, model)

	frame = cv2.resize(frame, new_shape, interpolation = cv2.INTER_LINEAR)
	# mask  = cv2.resize(mask, new_shape, interpolation = cv2.INTER_LINEAR)
	mask  = cv2.resize(mask, new_shape, interpolation = cv2.INTER_NEAREST)

	frame[np.where((mask != 0))] = (0,0,255)
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

	return mask, frame


def execute_model():

	if args.pic:
		frame = cv2.imread(args.filepath)
		if frame is None:
			print('Failed to open file')
			exit(1)

		# f_height, f_width, f_cahnnels = frame.shape
		# scale_x = float(f_width)/nn_out_size
		# scale_y = float(f_height)/nn_out_size

		model = get_unet()
		model.load_weights(args.weights)

		mask, frame = mark_image(frame, model)
		frame = np.hstack((mask,frame))

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

		model = get_unet()
		model.load_weights(args.weights)

		if args.fps:
			mask_obtain_time = 0
			num_frames = 120
			start = time.time()
			for i in xrange(0, num_frames) :
				ret, frame = cap.read()
				if frame is None:
					exit(1)

				start_mask = time.time()

				mask, frame = mark_image(frame, model)

				mask_obtain_time += (time.time() - start_mask)
				
				frame = np.hstack((mask,frame))
				cv2.imshow('frame',frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			seconds = time.time() - start
			print('Mean time per frame : {0} ms'.format(seconds / num_frames * 1000))
			print('Estimated frames per second : {0}'.format(num_frames / seconds))
			print('Bbox obtain mean time : {0} ms'.format(mask_obtain_time / num_frames * 1000))

		else:	
			while(cap.isOpened()):
				ret, frame = cap.read()
				if frame is None:
					exit(1)

				mask, frame = mark_image(frame, model)
				frame = np.hstack((mask,frame))
				
				cv2.imshow('frame',frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

		cap.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	execute_model()
