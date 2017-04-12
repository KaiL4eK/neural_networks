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

def preprocess_regress(imgs, bboxes):
	imgs_p   = np.ndarray((imgs.shape[0],  nn_img_side, nn_img_side, 3), dtype=np.float32)
	bboxes_p = np.ndarray((bboxes.shape[0], 4), 						 dtype=np.float32)

	# scale_x = float(nn_img_side)/npy_img_width
	# scale_y = float(nn_img_side)/npy_img_height

	# Percentage of real size
	scale_x = 1/float(npy_img_width)
	scale_y = 1/float(npy_img_height)

	for i in range(imgs.shape[0]):
		imgs_p[i]   = preprocess_img(imgs[i])
		bboxes_p[i] = np.multiply(bboxes[i], [scale_x, scale_y, scale_x, scale_y])

	return imgs_p, bboxes_p


print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)
imgs_train, imgs_bbox_train = load_train_data()
imgs_train, imgs_bbox_train = preprocess_regress(imgs_train, imgs_bbox_train)



class TestCallback(Callback):
	def __init__(self):
		pass

	def on_epoch_end(self, epoch, logs={}):
		total = imgs_train.shape[0]
		i_image = int(random.random() * total)

		true_bbox = imgs_bbox_train[i_image]
		print(true_bbox)
		bbox = self.model.predict(np.reshape(imgs_train[i_image], (1, nn_img_side, nn_img_side, 3)), verbose=0)
		bbox = bbox[0]
		print(bbox)

		xA = max(bbox[0], true_bbox[0])
		yA = max(bbox[1], true_bbox[1])
		xB = min(bbox[2], true_bbox[2])
		yB = min(bbox[3], true_bbox[3])

		intersection = (xB - xA) * (yB - yA)
		boxAArea = (bbox[2] - bbox[0]) 			 * (bbox[3] - bbox[1])
		boxBArea = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
		loss = intersection / (boxAArea + boxBArea - intersection)

		print('\nTesting loss: {}'.format(1 - loss))

		eval_loss = self.model.evaluate(np.reshape(imgs_train[i_image], (1, nn_img_side, nn_img_side, 3)), 
										np.reshape(imgs_bbox_train[i_image], (1, 4)), batch_size=1, verbose=0)
		print('Eval loss: {}\n'.format(eval_loss))

		if loss < 0:
			print('Warning!!!!<<<<<<<')


def train_regression():
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	model = regression_model()
	print_summary(model)
	# plot_model(model, show_shapes=True)

	if args.weights:
		model.load_weights(args.weights)

	print('-'*30)
	print('Fitting model...')
	print('-'*30)

	model.fit(imgs_train, imgs_bbox_train, batch_size=8, epochs=1000, verbose=1, shuffle=True, validation_split=0.11, 
				callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1), TestCallback()])

if __name__ == '__main__':
	train_regression()
