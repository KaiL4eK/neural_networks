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

nn_img_side = 196

class_list = ['forward', 'right', 'left', 'forward and right', 'brick', 'stop']
num_classes = len(class_list)

# Output is resized, BGR, mean subtracted, [0, 1.] scaled by values
def preprocess_img(img):
	img = cv2.resize(img, (nn_img_side, nn_img_side), interpolation = cv2.INTER_LINEAR)
	img = img.astype('float32', copy=False)
	img /= 255.
	return img

def classification_model():

	input = Input(shape=(nn_img_side, nn_img_side, 3))

	conv1 = Conv2D(32,(3,3),activation='elu',padding='same')(input)
	# conv1 = Conv2D(32,(3,3),activation='relu',padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(3, 3))(conv1)
	drop1 = Dropout(0.25)(pool1)

	conv2 = Conv2D(64,(3,3),activation='elu',padding='same')(drop1)
	conv2 = Conv2D(64,(3,3),activation='elu',padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(3, 3))(conv2)
	drop2 = Dropout(0.25)(pool2)

	conv3 = Conv2D(128,(3,3),activation='elu',padding='same')(drop2)
	conv3 = Conv2D(128,(3,3),activation='elu',padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	drop3 = Dropout(0.25)(pool3)

	conv4 = Conv2D(256,(3,3),activation='elu',padding='same')(drop3)
	# conv4 = Conv2D(256,(3,3),activation='relu',padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	drop4 = Dropout(0.25)(pool4)

	# conv5 = Conv2D(512,(3,3),activation='relu',padding='same')(drop4)
	# conv5 = Conv2D(512,(3,3),activation='relu',padding='same')(conv5)
	# pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
	# drop5 = Dropout(0.25)(pool5)

	flat  = Flatten()(drop4)

	fc_cls      = Dense(1024, activation='tanh')(flat)
	drop_fc_cls = Dropout(0.5)(fc_cls)
	out_cls     = Dense(num_classes, activation='sigmoid', name='out_cls')(drop_fc_cls)

	model = Model(input, out_cls)
	model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=[])

	return model

def get_network_model():
	model = classification_model()

	print_summary(model)
	plot_model(model, show_shapes=True)

	return model	
