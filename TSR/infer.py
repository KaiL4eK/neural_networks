#! /usr/bin/env python

import os
import json
import cv2
import time
import data
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from tqdm import tqdm
import numpy as np

import argparse
argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-c', '--conf', help='path to configuration file')
argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')
argparser.add_argument('-w', '--weights', help='weights path')
argparser.add_argument('-o', '--output', default='output/', help='path to output directory')

args = argparser.parse_args()


def _main_():
    config_path = args.conf
    input_path = args.input
    weights_path = args.weights

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    classes = data.get_classes(config['train']['cache_name'])

    if not classes:
        print('Failed to get train classes')

    infer_model = load_model(weights_path)

    image_paths = []

    if os.path.isdir(input_path):
        for root, subdirs, files in os.walk(input_path):
            pic_extensions = ('.jpg', '.png', 'JPEG', '.ppm')
            image_paths += [os.path.join(root, file) for file in files if file.endswith(pic_extensions)]
    else:
        image_paths += [input_path]

    processing_count = 0
    sum_time = 0

    # the main loop
    for image_path in tqdm(image_paths):
        src_image = cv2.imread(image_path)
        # print(image_path)

        start_time = time.time()

        net_input_shape = (config['model']['input_side_sz'],
                           config['model']['input_side_sz'])

        image = cv2.resize(src_image, net_input_shape)

        image = data.normalize(image)
        image = np.expand_dims(image, axis=0)
        result = infer_model.predict(image)[0]

        sum_time += time.time() - start_time
        processing_count += 1

        max_idx = np.argmax(result)
        print(classes[max_idx], max_idx)

        cv2.imshow('1', src_image)
        if 27 == cv2.waitKey(0):
            break

    fps = processing_count / sum_time
    print('Result: {}'.format(fps))


if __name__ == '__main__':
    _main_()
