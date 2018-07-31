import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_gpu() 

#Size of images
IMAGE_INPUT_WIDTH = 240
IMAGE_INPUT_HEIGHT = 240

def prepareInputImage(img, img_width=IMAGE_INPUT_WIDTH, img_height=IMAGE_INPUT_HEIGHT):
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = img.transpose((2,0,1))

    return np.uint8(img)

#Read model architecture and trained model's weights
net = caffe.Net('deploy.prototxt', 'weights', caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 1./255)

'''
Making predicitions
'''
##Reading image paths
PATH = '../Trafic_road_Training_db/Initial_db/kostya_signs'

test_img_paths = [img_path for img_path in glob.glob(PATH + '/*.png')]

test_ids = []
preds = []

#Making predictions
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = prepareInputImage(img, img_width=IMAGE_INPUT_WIDTH, img_height=IMAGE_INPUT_WIDTH)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    net.forward()

    img_in = net.blobs['data'].data[0,:] * 255
    img_in = img_in.transpose((1,2,0))
    img_in = np.uint8(img_in)
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

    img_res = net.blobs['score'].data[0,0]
    img_res = np.uint8(img_res)

    cv2.imshow('in', img_in)
    cv2.imshow('res', img_res)

    if cv2.waitKey(0) == 27:
        break
