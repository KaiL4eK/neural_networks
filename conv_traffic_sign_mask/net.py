from keras.models import Sequential
from keras.losses import binary_crossentropy, mean_squared_error
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout, Deconv2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.layer_utils import print_summary
from keras.utils.vis_utils import plot_model
import numpy as np
import cv2

import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

nn_img_side = 400
nn_out_size = 100

### Rates ###
# 27.5 ms - processing time gpu
# 

def intersect_over_union(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)

	intersection = K.sum(y_true_f * y_pred_f)
	union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
	# return K.switch( K.equal(union, 0), K.variable(1), intersection / union) 
	return K.mean(intersection / union) 

def iou_loss(y_true, y_pred):
	return 1 - intersect_over_union(y_true, y_pred)

def loss_empower(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)

	eps = 1e-10

	x = y_pred_f
	# x = K.clip(y_pred_f, eps, 1)
	z = K.round(y_true_f)

	result_true  = tf.multiply(z, K.log(x))
	result_false = tf.multiply((1 - z), K.log(1 - x))

	score_false_negative = tf.where(K.equal(x, 0), z, K.zeros_like(z))

	# return K.sum(-result_true) * 1e-3 + iou_loss(y_true, y_pred)

	return K.sum(score_false_negative) * 1e-2 + iou_loss(y_true, y_pred)


def full_loss(y_true, y_pred):
	return binary_crossentropy(y_true, y_pred) + binary_crossentropy_empower(y_true, y_pred) + iou_loss(y_true, y_pred)

### Image preprocessing ###

# Output is resized, BGR, mean subtracted, [0, 1.] scaled by values
def preprocess_img(img):
	img = cv2.resize(img, (nn_img_side, nn_img_side), interpolation = cv2.INTER_LINEAR)

	# img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	# img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	# img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	img = img.astype('float32', copy=False)
	# img[:,:,0] -= 103.939
	# img[:,:,1] -= 116.779
	# img[:,:,2] -= 123.68
	img /= 255.

	return img

def preprocess_mask(img):
	img = cv2.resize(img, (nn_out_size, nn_out_size), interpolation = cv2.INTER_NEAREST)
	img = img.astype('float32', copy=False)
	img /= 255.

	return img

def get_unet(lr=1e-3):
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='elu', padding='same', input_shape=(nn_img_side, nn_img_side, 3)))
	# model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
	# model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
	# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
	model.add(Dropout(0.25))
	# model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
	# model.add(Dropout(0.25))
	model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
	model.add(Dropout(0.25))
	model.add(Conv2D(1, (3, 3), activation='hard_sigmoid', padding='same'))

	# model.compile(optimizer='adadelta', loss=iou_loss, metrics=[binary_crossentropy])
	model.compile(optimizer=Adam(lr=lr), loss=loss_empower, metrics=[binary_crossentropy, loss_empower, iou_loss])
	
	print_summary(model)
	# plot_model(model, show_shapes=True)

	return model
