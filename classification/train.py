from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, Callback, RemoteMonitor
import random

from data import *
from net import *
import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-w', '--weights', action='store',      help='Path to weights file')
parser.add_argument('-t', '--test',    action='store_true', help='Test model for loss')

args = parser.parse_args()

def preprocess_regress(imgs, classes):
	imgs_p   = np.ndarray((imgs.shape[0],  nn_img_side, nn_img_side, 3), dtype=np.float32)
	class_p  = np.ndarray((classes.shape[0], num_classes), 			     dtype=np.float32)

	for i in range(imgs.shape[0]):
		imgs_p[i]   = preprocess_img(imgs[i])
		class_p[i]  = np_utils.to_categorical(class_list.index(classes[i]), num_classes)	
		# print(class_p[i], class_p[i].shape)

	return imgs_p, class_p


print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

imgs_train 			= npy_data_load_images()
imgs_class_train 	= npy_data_load_classes()
imgs_train, imgs_class_train = preprocess_regress(imgs_train, imgs_class_train)

def train_regression():
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	model = get_network_model()

	if args.weights:
		model.load_weights(args.weights)

	if args.test:
		i_image = 1 #int(random.random() * total)
		img 			= np.reshape(imgs_train[i_image], (1, nn_img_side, nn_img_side, 3))
		true_class 		= np.reshape(imgs_class_train[i_image], (1, num_classes))
		
		input_data  = img
		output_data = true_class

		class_rates = model.predict(input_data, verbose=0)
		print(class_rates)
		print(output_data)
		
		eval_loss = model.evaluate(input_data, output_data)
		print('Eval loss:\t{}\n'.format(eval_loss))

		cv2.imshow('frame', imgs_train[i_image])
		cv2.waitKey(0)
	else:
		print('-'*30)
		print('Fitting model...')
		print('-'*30)

		input_data  = imgs_train
		output_data = imgs_class_train

		model.fit(input_data, output_data, batch_size=50, epochs=7000, verbose=1, shuffle=True, validation_split=0,
					callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

if __name__ == '__main__':
	train_regression()
