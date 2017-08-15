from __future__ import print_function

import sys
sys.path.append('..')

try:
    xrange
except NameError:
    xrange = range

# stop
# (0.8, 0.8888888888888888)
# 0.842105263158
# pedestrian
# (1.0, 0.9090909090909091)
# 0.952380952381
# main_road
# (0.8, 0.8888888888888888)
# 0.842105263158
# bus_stop
# (0.8666666666666667, 1.0)
# 0.928571428571

import os
import cv2
import numpy as np
import time
from keras.models import Model, load_model, save_model

from labeled_traffic_sign import net
# from common import confusion_matrix as cm
# import argparse

# parser = argparse.ArgumentParser(description='Process test with ANN')
# parser.add_argument('weights', action='store', help='Path to weights file')
# parser.add_argument('filepath', action='store', help='Path to video file to process')
# parser.add_argument('-t', '--threshold', action='store', default='.8', help='Threshold value')
# parser.add_argument('-p', '--pic', action='store_true', help='Process picture')
# parser.add_argument('-n', '--negative', action='store_true', help='Negative creation')

# args = parser.parse_args()

data_path = '../raw_data/car_register_test'

module_delimiter        = '-'
module_data_delimiter   = ','
extension_delimiter     = '.'

id_module_idx           = 0
frame_number_module_idx = 1
bbox_module_idx         = 2
label_module_idx        = 3

model = net.get_network_model(0)
model.load_weights('../labeled_traffic_sign/weights_best.h5')

if os.path.exists('result.txt'):
    os.remove('result.txt')

def test_label_model():

    paths = (os.path.join(root, filename)
            for root, _, filenames in os.walk(data_path)
            for filename in filenames if filename.endswith('.png'))

    for i, image_path in enumerate(paths):
        # Ground truth creation
        image_name = os.path.basename(image_path)

        if len(image_name.split(extension_delimiter)) != 2:
            print('File extension not found')
            print(image_name)
            exit(1)

        info = image_name.split(extension_delimiter)[0]
        module_list = info.split(module_delimiter)

        if len(module_list) != 4:
            print('Modules info broken')
            print(image_name)
            exit(1)

        module_label_list = module_list[label_module_idx].split(module_data_delimiter)

        image_label_vector = [0] * len(net.glob_label_list)
        for label in module_label_list:
            if label in net.glob_label_list:
                label_list_idx = net.glob_label_list.index(label)
                image_label_vector[label_list_idx] = 1

        # Image opening

        image = cv2.imread(image_path)
        if image is None:
            print('Failed to open file')
            exit(1)

        image_p   = net.preprocess_img(image)

        nn_input            = image_p
        nn_output_truth     = image_label_vector

        nn_output_pred = model.predict(np.array([nn_input]))[0]

        with open('result.txt', 'a') as file:
            file.write('{} / {}\n'.format( nn_output_truth, np.array(nn_output_pred, dtype=float) ))

        print(nn_output_truth, nn_output_pred)

        cv2.imshow('1', image_p)
        cv2.waitKey(0)

if __name__ == '__main__':
    test_label_model()
