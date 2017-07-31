from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, Callback, RemoteMonitor
import random

from data import *
import net
import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-b', '--batch_size', default=1, action='store', help='Size of batch to learn')
parser.add_argument('-w', '--weights', action='store',      help='Path to weights file')
parser.add_argument('-t', '--test',    action='store_true', help='Test model for loss')

args = parser.parse_args()

def preprocess_classification(imgs, label_list):
	imgs_p   = np.ndarray((imgs.shape[0],  net.nn_img_side, net.nn_img_side, 3), 	dtype=np.float32)
	label_p  = np.ndarray((label_list.shape[0], len(net.glob_label_list)),	dtype=np.float32)

	for i in range(imgs.shape[0]):
		imgs_p[i]   = net.preprocess_img(imgs[i])
		label_p[i] 	= label_list[i]
		# print(label_p[i], label_p[i].shape)

	return imgs_p, label_p


print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

imgs_train 			= npy_data_load_images()
imgs_class_train 	= npy_data_load_classes()
imgs_train, imgs_class_train = preprocess_classification(imgs_train, imgs_class_train)

def train_classification():
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	model = net.get_network_model(1e-5)

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

		model.fit(input_data, output_data, batch_size=int(args.batch_size), epochs=7000, verbose=1, shuffle=True, validation_split=0.2,
					callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

if __name__ == '__main__':
	train_classification()
