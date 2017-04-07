from keras.models import Sequential
from keras.losses import binary_crossentropy, mean_squared_error
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout, Deconv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.layer_utils import print_summary
import numpy as np

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_side_size = 96
mean = 116.357

smooth = 1.

def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

def intersect_on_union(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)

	intersection = K.sum(y_true_f * y_pred_f)
	return intersection / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection)

def iou_loss(y_true, y_pred):
	return 1-intersect_on_union(y_true, y_pred)

def shape_metrics(y_true, y_pred):
	# y_true_f = K.flatten(y_true)
	return K.shape(y_true)[0]

def get_unet():
	# inputs = Input((img_side_size, img_side_size, 3))
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_side_size, img_side_size, 3)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.25))

	# up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.5))

	# up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.5))

	# up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

	# up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

	model.add(Conv2D(1, (1, 1), activation='hard_sigmoid'))

	# model = Model(inputs=[inputs], outputs=[conv10])
	

	# model.compile(optimizer=Adam(lr=1e-5), loss=binary_crossentropy, metrics=[dice_coef, intersect_on_union])
	model.compile(optimizer=Adam(lr=1e-5), loss=iou_loss, metrics=[])
	# model.compile(optimizer='sgd', loss=iou_loss, metrics=[dice_coef])
	print_summary(model)

	return model


def intersect_on_union_bbox(y_true, y_pred):
	# y_true_f = K.flatten(y_true)
	# y_pred_f = K.flatten(y_pred)

	y_true_f = y_true
	y_pred_f = y_pred

	K.print_tensor(y_true)

	xA = K.maximum(y_true_f[:, 0], y_pred_f[:, 0])
	yA = K.maximum(y_true_f[:, 1], y_pred_f[:, 1])
	xB = K.minimum(y_true_f[:, 2], y_pred_f[:, 2])
	yB = K.minimum(y_true_f[:, 3], y_pred_f[:, 3])

	dx = xA - xB
	dy = yA - yB

	intersection = dx * dy
	boxAArea = (y_true_f[:, 2]) * (y_true_f[:, 3])
	boxBArea = (y_pred_f[:, 2]) * (y_pred_f[:, 3])

	return intersection / (boxAArea + boxBArea - intersection)

def iou_loss_bbox(y_true, y_pred):
	return 1-intersect_on_union_bbox(y_true, y_pred)


def regression_model():
	inputs = Input((img_side_size, img_side_size, 3))

	conv1 = Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64,(3,3),activation='relu',padding='same')(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3_1 = Conv2D(64,(3,3),activation='relu',padding='same')(pool2)
	conv3_2 = Conv2D(128,(3,3),activation='relu',padding='same')(conv3_1)
	conv3_3 = Conv2D(128,(3,3),activation='relu',padding='same')(conv3_2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_3)  

	conv4_1 = Conv2D(256,(3,3),activation='relu',padding='same')(pool3)
	conv4_2 = Conv2D(256,(3,3),activation='relu',padding='same')(conv4_1)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)    

	conv5_1 = Conv2D(256,(3,3),activation='relu',padding='same')(pool4)
	conv5_2 = Conv2D(256,(3,3),activation='relu',padding='same')(conv5_1)
	pool5 = MaxPooling2D(pool_size=(2, 2))(conv5_2)

	conv6_1 = Conv2D(256,(3,3),activation='relu',padding='same')(pool5)
	conv6_2 = Conv2D(256,(3,3),activation='relu',padding='same')(conv6_1)
	pool6 = MaxPooling2D(pool_size=(2, 2))(conv6_2)

	flat = Flatten()(pool6)
	bbox_pred  	= Dense(1024,activation='softmax')(flat)
	bbox  		= Dense(4,activation='softmax')(bbox_pred)

	model = Model(inputs=[inputs], outputs=[bbox])
	print_summary(model)

	model.compile(optimizer=Adam(lr=1e-5), loss=iou_loss_bbox, metrics=[intersect_on_union_bbox])

	return model
