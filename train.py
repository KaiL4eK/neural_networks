from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
import random

from data import load_train_data, load_test_data, data_path
from net import *
import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-w', '--weights', action='store', help='Path to weights file')

args = parser.parse_args()

class TestCallback(Callback):
	def __init__(self):
		pass

	def on_epoch_end(self, epoch, logs={}):
		images = os.listdir(data_path)
		total = len(images)
		i_image = int(random.random() * total)

		test_img = cv2.imread(os.path.join(data_path, images[i_image]))
		height, width, channels = test_img.shape
		cv2.imshow('1', test_img)
		test_img = preprocess_img(test_img)
		
		
		info = images[i_image].split(';')
		ul_x = max(0, int(info[1]))
		ul_y = max(0, int(info[2]))
		lr_x = ul_x + max(0, int(info[3]))
		lr_y = ul_y + max(0, int(info[4]))
		img_mask_truth = np.zeros((240, 320), np.uint8)
		cv2.rectangle(img_mask_truth, (ul_x, ul_y), (lr_x, lr_y), thickness=-1, color=255 )

		imgs_mask = self.model.predict(np.reshape(test_img, (1, nn_img_side, nn_img_side, 3)), verbose=0)
		# img_mask = cv2.resize(imgs_mask[0], (width, height), interpolation = cv2.INTER_NEAREST)

		cv2.imshow('2', imgs_mask[0])

		img_mask_truth = cv2.resize(img_mask_truth, (nn_img_side, nn_img_side), interpolation = cv2.INTER_NEAREST)
		img_mask_truth = img_mask_truth.astype('float32')
		img_mask_truth /= 255.
		
		cv2.imshow('3', img_mask_truth)
		if cv2.waitKey(100) == 27:
			cv2.destroyAllWindows()
			exit(1)

		loss = self.model.evaluate( np.reshape(test_img, 	   (1, nn_img_side, nn_img_side, 3)), 
									np.reshape(img_mask_truth, (1, nn_img_side, nn_img_side, 1)), verbose=0, batch_size=1)
		
		# input_data = self.model.layers[0].get_output()
		# output_data = self.model.layers[31].get_output()
		# print(input_data.shape, output_data.shape)

		print('\nTesting loss: {}\n'.format(loss))



def preprocess_arrays(imgs, masks):
	imgs_p  = np.ndarray((imgs.shape[0],  nn_img_side, nn_img_side, 3), dtype=np.float32)
	masks_p = np.ndarray((masks.shape[0], nn_img_side, nn_img_side),    dtype=np.float32)
	for i in range(imgs.shape[0]):
		imgs_p[i]  = preprocess_img(imgs[i])
		masks_p[i] = preprocess_mask(masks[i])
		
		# cv2.imshow('1', imgs_p[i])
		# cv2.imshow('2', masks_p[i])
		# if cv2.waitKey(0) == 27:
		# 	exit(1)

	return imgs_p, masks_p[..., np.newaxis]


def train_and_predict():
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, imgs_mask_train = load_train_data()
	imgs_train, imgs_mask_train = preprocess_arrays(imgs_train, imgs_mask_train)

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	model = get_unet()
	print_summary(model)
	# plot_model(model, show_shapes=True)

	if args.weights:
		model.load_weights(args.weights)

	print('-'*30)
	print('Fitting model...')
	print('-'*30)

	model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=2000, verbose=1, shuffle=True, validation_split=0.11, 
				callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

if __name__ == '__main__':
	train_and_predict()
