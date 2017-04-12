from keras.models import Sequential
from keras.losses import binary_crossentropy, mean_squared_error, hinge
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout, Deconv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.layer_utils import print_summary
from keras.utils.vis_utils import plot_model
import numpy as np
import cv2

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

nn_img_side = 144

# Output is resized, BGR, mean subtracted, [0, 1.] scaled by values
def preprocess_img(img):
	img = cv2.resize(img, (nn_img_side, nn_img_side), interpolation = cv2.INTER_LINEAR)
	# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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

	# pred_lr_x = K.clip(pred_lr_x, 0, 1)
	# pred_lr_y = K.clip(pred_lr_y, 0, 1)

	# pred1 = K.less_equal( pred_lr_x, pred_ul_x )
	# pred2 = K.less_equal( pred_lr_y, pred_ul_y )
	# sess = K.get_session()
	# if sess.run(pred1) or sess.run(pred2):
		# return K.variable(0);

	xA = K.maximum(y_true_f[:, 0], pred_ul_x)
	yA = K.maximum(y_true_f[:, 1], pred_ul_y)
	xB = K.minimum(y_true_f[:, 0] + y_true_f[:, 2], pred_lr_x)
	yB = K.minimum(y_true_f[:, 1] + y_true_f[:, 3], pred_lr_y)

	intersection = (xB - xA) * (yB - yA)
	boxAArea = (pred_lr_x - pred_ul_x) 			 * (pred_lr_y - pred_ul_y)
	boxBArea = (y_true_f[:, 2] - y_true_f[:, 0]) * (y_true_f[:, 3] - y_true_f[:, 1])

	# intersection = K.switch( pred1, intersection, intersection )
	# intersection = K.switch( pred2, K.variable(0), intersection )

	res = intersection / (boxAArea + boxBArea - intersection)
	return res

def iou_loss(y_true, y_pred):
	return 1-intersect_over_union_bbox(y_true, y_pred)

def check_shape_metrics(y_true, y_pred):
	return K.shape(y_true)[0]


def regression_model():
	model = Sequential()

	model.add(Conv2D(32,(3,3),activation='relu',padding='same', input_shape=(nn_img_side, nn_img_side, 3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
	model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
	model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512,activation='sigmoid'))
	model.add(Dropout(0.25))
	model.add(Dense(4,activation='sigmoid'))

	print_summary(model)
	model.compile(optimizer=Adam(lr=1e-5), loss=iou_loss, metrics=[])

	return model
