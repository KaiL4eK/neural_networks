import os
import numpy as np
from scipy import special
import cv2

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def init_gpu_session(rate):

    import keras.backend as K
    import tensorflow as tf

    config = tf.ConfigProto()

    config.gpu_options.allow_growth                     = True
    config.gpu_options.per_process_gpu_memory_fraction  = rate

    K.set_session(tf.Session(config=config))

def print_pretty(str):
	print('-'*30)
	print(str)
	print('-'*30)

def sigmoid(x):
    return special.expit(x)
    # return 1.0/(1.0 + np.exp(-x))
    # y = np.exp(x); return y/(1+y)
    # return x


def image_preprocess(img, result_sz):

    img = cv2.resize(img, result_sz[0:2])
    img = img / 255. #* 2 - 1

    return img

def mask_preprocess(img, result_sz):

    img = cv2.resize(img, result_sz, interpolation=cv2.INTER_NEAREST)
    img = img / 255. #* 2 - 1

    return img
