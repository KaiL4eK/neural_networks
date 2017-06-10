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

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

nn_img_side = 240
nn_out_size = 120

### Rates ###
# 27.5 ms - processing time gpu
# 

def intersect_over_union(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)

	intersection = K.sum(y_true_f * y_pred_f)
	union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
	# return K.switch( K.equal(union, 0), K.variable(1), intersection / union) 
	return intersection / union 

def iou_loss(y_true, y_pred):
	return 1 - intersect_over_union(y_true, y_pred)

### Image preprocessing ###

# Output is resized, BGR, mean subtracted, [0, 1.] scaled by values
def preprocess_img(img):
	img = cv2.resize(img, (nn_img_side, nn_img_side), interpolation = cv2.INTER_CUBIC)

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

### Net structure ###

def get_unet():
	model = Sequential()


	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(nn_img_side, nn_img_side, 3)))
	# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))
	
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	# model.add(MaxPooling2D(pool_size=(2, 2)))

	# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

	# model.add(UpSampling2D(size=(2, 2)))
	# model.add(Deconv2D(64, (2, 2), activation='relu', padding='same'))
	# model.add(Dropout(0.25))

	model.add(UpSampling2D(size=(2, 2)))
	model.add(Deconv2D(64, (3, 3), activation='relu', padding='same'))
	# model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(UpSampling2D(size=(2, 2)))
	model.add(Deconv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(UpSampling2D(size=(2, 2)))
	model.add(Deconv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Deconv2D(1, (1, 1), activation='hard_sigmoid'))


	# model.compile(optimizer='adadelta', loss=iou_loss, metrics=[binary_crossentropy])
	model.compile(optimizer=Adam(lr=1e-3), loss=iou_loss, metrics=[binary_crossentropy])
	

	print_summary(model)
	plot_model(model, show_shapes=True)

	return model
