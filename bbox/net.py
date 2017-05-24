from keras.models import Sequential, Model
from keras.losses import binary_crossentropy, mean_squared_error, hinge
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout, Deconv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.layer_utils import print_summary
from keras.utils.vis_utils import plot_model
import numpy as np
import cv2

import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

nn_img_side = 144

class_list = ['forward', 'right', 'left', 'forward and right', 'brick', 'stop']
num_classes = len(class_list)

# Output is resized, BGR, mean subtracted, [0, 1.] scaled by values
def preprocess_img(img):
	img = cv2.resize(img, (nn_img_side, nn_img_side), interpolation = cv2.INTER_LINEAR)
	img = img.astype('float32', copy=False)
	img[:,:,0] -= 103.939
	img[:,:,1] -= 116.779
	img[:,:,2] -= 123.68
	img /= 255.
	return img

def intersect_over_union_bbox(y_true, y_pred):

	y_true_f = y_true
	y_pred_f = y_pred

	pred_ul_x = y_pred_f[:, 0]
	pred_ul_y = y_pred_f[:, 1]
	pred_lr_x = y_pred_f[:, 0] + y_pred_f[:, 2]
	pred_lr_y = y_pred_f[:, 1] + y_pred_f[:, 3]

	true_ul_x = y_true_f[:, 0]
	true_ul_y = y_true_f[:, 1]
	true_lr_x = y_true_f[:, 0] + y_true_f[:, 2]
	true_lr_y = y_true_f[:, 1] + y_true_f[:, 3]

	true_width  = y_true_f[:, 2]
	true_height = y_true_f[:, 3]

	pred_width  = y_pred_f[:, 2]
	pred_height = y_pred_f[:, 3]

	pred_lr_x = K.clip(pred_lr_x, 0, 1)
	pred_lr_y = K.clip(pred_lr_y, 0, 1)

	xA = K.maximum(true_ul_x, pred_ul_x)	# Left
	yA = K.maximum(true_ul_y, pred_ul_y)	# Up
	xB = K.minimum(true_lr_x, pred_lr_x)	# Right
	yB = K.minimum(true_lr_y, pred_lr_y)	# Down

	boxAArea = pred_width * pred_height
	boxBArea = true_width * true_height

	xNotIntersect = tf.logical_or(K.greater_equal(true_ul_x, pred_lr_x), K.greater_equal(pred_ul_x, true_lr_x))
	yNotIntersect = tf.logical_or(K.greater_equal(true_ul_y, pred_lr_y), K.greater_equal(pred_ul_y, true_lr_y))
	notIntersect  = tf.logical_or(xNotIntersect, yNotIntersect)

	intersection = tf.multiply((xB - xA), (yB - yA))

	fault_ul_square = K.abs(tf.multiply((true_ul_x - pred_ul_x), (true_ul_y - pred_ul_y)))
	fault_lr_square = K.abs(tf.multiply((true_lr_x - pred_lr_x), (true_lr_y - pred_lr_y)))

	squares_dist = K.clip((fault_ul_square + fault_lr_square + boxAArea + boxBArea) * 3, 0, 1)

#	res = tf.where(notIntersect, (K.abs(intersection) * -1) / (boxAArea + boxBArea), intersection / (boxAArea + boxBArea - intersection));
	res = tf.where(notIntersect, 1-squares_dist, intersection / (boxAArea + boxBArea - intersection))

#	isNegativeSample = K.equal(true_ul_x, true_ul_y)
#	negativeSamplesRate = 5
#	res = tf.where(isNegativeSample, K.ones_like(res) - K.clip(K.maximum(pred_lr_x, pred_lr_y) * negativeSamplesRate, 0, 1), res)
	
	return K.mean(res)

def iou_loss(y_true, y_pred):
	return 1-intersect_over_union_bbox(y_true, y_pred)
	# return intersect_over_union_bbox(y_true, y_pred)

def check_shape_metrics(y_true, y_pred):
	return K.shape(y_true)[0]

def two_output_model():

	input = Input(shape=(nn_img_side, nn_img_side, 3))

	conv1 = Conv2D(32,(3,3),activation='relu',padding='same')(input)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	drop1 = Dropout(0.25)(pool1)

	conv2 = Conv2D(64,(3,3),activation='relu',padding='same')(drop1)
	conv2 = Conv2D(64,(3,3),activation='relu',padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	drop2 = Dropout(0.25)(pool2)

	conv3 = Conv2D(128,(3,3),activation='relu',padding='same')(drop2)
	conv3 = Conv2D(128,(3,3),activation='relu',padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	drop3 = Dropout(0.25)(pool3)

	conv4 = Conv2D(256,(3,3),activation='relu',padding='same')(drop3)
	conv4 = Conv2D(256,(3,3),activation='relu',padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	drop4 = Dropout(0.25)(pool4)

	conv5 = Conv2D(512,(3,3),activation='relu',padding='same')(drop4)
	pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
	drop5 = Dropout(0.25)(pool5)
	flat  = Flatten()(drop5)

	fc_bb      = Dense(1024, activation='relu')(flat)
	drop_fc_bb = Dropout(0.5)(fc_bb)
	out_bb     = Dense(4, activation='sigmoid', name='out_bb')(drop_fc_bb)

	fc_cls      = Dense(1024, activation='relu')(flat)
	drop_fc_cls = Dropout(0.5)(fc_cls)
	out_cls     = Dense(num_classes, activation='softmax', name='out_cls')(drop_fc_cls)
	
	model = Model(input, [out_bb, out_cls])
	model.compile(optimizer=Adam(lr=1e-5), loss={'out_bb': iou_loss, 'out_cls': 'categorical_crossentropy'}, metrics=[])

	return model

def sequential_functional_model():

	input = Input(shape=(nn_img_side, nn_img_side, 3))

	conv1 = Conv2D(32,(3,3),activation='relu',padding='same')(input)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	drop1 = Dropout(0.25)(pool1)

	conv2 = Conv2D(64,(3,3),activation='relu',padding='same')(drop1)
	conv2 = Conv2D(64,(3,3),activation='relu',padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	drop2 = Dropout(0.25)(pool2)

	conv3 = Conv2D(128,(3,3),activation='relu',padding='same')(drop2)
	conv3 = Conv2D(128,(3,3),activation='relu',padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	drop3 = Dropout(0.25)(pool3)

	conv4 = Conv2D(256,(3,3),activation='relu',padding='same')(drop3)
	conv4 = Conv2D(256,(3,3),activation='relu',padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	drop4 = Dropout(0.25)(pool4)

	conv5 = Conv2D(512,(3,3),activation='relu',padding='same')(drop4)
	pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
	drop5 = Dropout(0.25)(pool5)
	flat  = Flatten()(drop5)

	fc_bb      = Dense(1024, activation='relu')(flat)
	drop_fc_bb = Dropout(0.5)(fc_bb)
	out_bb     = Dense(4, activation='sigmoid')(drop_fc_bb)

	model = Model(input, out_bb)
	model.compile(optimizer=Adam(lr=1e-5), loss='iou_loss', metrics=[])

	return model

def sequential_model():
	model = Sequential()

	model.add(Conv2D(32,(3,3),activation='relu',padding='same', input_shape=(nn_img_side, nn_img_side, 3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
	# model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
	model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	# model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Dropout(0.25))

	# model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Dropout(0.25))

	# model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	# model.add(Dense(512,activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(1024,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4,activation='sigmoid'))

	model.compile(optimizer=Adam(lr=1e-5), loss=iou_loss, metrics=[])

	return model
	

def get_network_model():
	# model = sequential_model()
	# model = sequential_functional_model()
	model = two_output_model()

	print_summary(model)
	plot_model(model, show_shapes=True)

	return model	
