'''
Title           :make_predictions_1.py
Description     :This script makes predictions using the 1st trained model and generates a submission file.
Author          :Adil Moujahid
Date Created    :20160623
Date Modified   :20160625
version         :0.2
usage           :python make_predictions_1.py
python_version  :2.7.11
'''

import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

# caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean images
# mean_blob = caffe_pb2.BlobProto()
# with open('../input/mean.binaryproto') as f:
#     mean_blob.ParseFromString(f.read())
# mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
#     (mean_blob.channels, mean_blob.height, mean_blob.width))

weights = 'caffenet_train_iter_35500.caffemodel'

#Read model architecture and trained model's weights
net = caffe.Net('deploy.prototxt', weights, caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_data_path = 'input/*jpg'
hidden_data_path = 'output'
test_img_paths = [img_path for img_path in glob.glob(test_data_path)]

#Making predictions

for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    tr_img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', tr_img)
    out = net.forward()
    pred_probas = out['prob']

    if pred_probas.argmax() == 0:
        print "Cat"
    else:
        print "Dog"

    batch_size, channels, height, width = net.blobs['conv1'].data.shape
    for i in range(channels):
        image = net.blobs['conv1'].data[0,i]
        # print image.shape
        # image = np.reshape(image, (height, width, 1))
        # print image.shape
        cv2.imwrite(hidden_data_path + '/conv1_' + str(i) + '.png', image)
    # img_out = net.blobs['label'].data[0,:] * 255
    # img_out = img_out.transpose((1,2,0))
    # img_out = np.uint8(img_out)

    input_img = net.blobs['data'].data[0,:]
    input_img = input_img.transpose((1,2,0))

    cv2.imshow('inp',np.uint8(input_img))
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
