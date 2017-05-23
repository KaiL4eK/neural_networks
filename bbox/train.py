from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, RemoteMonitor
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
		img = imgs_train[i_image]
		true_bbox = imgs_bbox_train[i_image]
		print(true_bbox)
		bbox = self.model.predict(np.reshape(img, (1, nn_img_side, nn_img_side, 3)), verbose=0)
		bbox = bbox[0]
		print(bbox)

		pred_ul_x = bbox[0]
		pred_ul_y = bbox[1]
		pred_lr_x = bbox[0] + bbox[2]
		pred_lr_y = bbox[1] + bbox[3]

		true_ul_x = true_bbox[0]
		true_ul_y = true_bbox[1]
		true_lr_x = true_bbox[0] + true_bbox[2]
		true_lr_y = true_bbox[1] + true_bbox[3]

		pred_lr_x = np.clip(pred_lr_x, 0, 1)
		pred_lr_y = np.clip(pred_lr_y, 0, 1)

		xA = max(pred_ul_x, true_ul_x)
		yA = max(pred_ul_y, true_ul_y)
		xB = min(pred_lr_x, true_lr_x)
		yB = min(pred_lr_y, true_lr_y)

		xIntersect = (true_ul_x - pred_lr_x) * (pred_ul_x - true_lr_x)
		xIntersect = np.clip( xIntersect * 10e9, 0, 1 )

		yIntersect = (true_ul_y - pred_lr_y) * (pred_ul_y - true_lr_y)
		yIntersect = np.clip( yIntersect * 10e9, 0, 1 )

		intersection = (xB - xA) * (yB - yA) * (xIntersect * yIntersect)
		boxAArea = (pred_lr_x - pred_ul_x) * (pred_lr_y - pred_ul_y)
		boxBArea = (true_lr_x - true_ul_x) * (true_lr_y - true_ul_y)
		loss = intersection / (boxAArea + boxBArea - intersection)
		print('\nTesting loss:\t{}'.format(1 - loss))

		eval_loss = self.model.evaluate(np.reshape(img, (1, nn_img_side, nn_img_side, 3)), 
										np.reshape(true_bbox, (1, 4)), batch_size=1, verbose=0)
		print('Eval loss:\t{}\n'.format(eval_loss))

		ul_x = int(pred_ul_x * nn_img_side)
		ul_y = int(pred_ul_y * nn_img_side)
		lr_x = int(pred_lr_x * nn_img_side)
		lr_y = int(pred_lr_y * nn_img_side)
		cv2.rectangle(img, (ul_x, ul_y), (lr_x, lr_y), thickness=2, color=(255, 0, 0) )
		ul_x = int(true_bbox[0] * nn_img_side)
		ul_y = int(true_bbox[1] * nn_img_side)
		lr_x = int((true_bbox[0]+true_bbox[2]) * nn_img_side)
		lr_y = int((true_bbox[1]+true_bbox[3]) * nn_img_side)
		cv2.rectangle(img, (ul_x, ul_y), (lr_x, lr_y), thickness=2, color=(0, 0, 255) )
		cv2.imshow('1', img)
		if cv2.waitKey(1) == 27:
			exit(1)

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

	remote = RemoteMonitor(root='http://localhost:9000')

	model.fit(imgs_train, imgs_bbox_train, batch_size=10, epochs=3000, verbose=1, shuffle=True, validation_split=0.11, 
				callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1), 
						   # remote,
						   # TestCallback()
						   ])

if __name__ == '__main__':
	train_regression()
