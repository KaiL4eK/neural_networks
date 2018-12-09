#! /usr/bin/env python

from TRTengine import TRTengine

from tqdm import tqdm
import os
import cv2
import argparse
import json
import time
from utils.bbox import draw_boxes
import numpy as np

argparser = argparse.ArgumentParser(
    description='TensorRT based inference for network')

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-i', '--input', help='path to an image or an video (mp4 format)', default=None)
argparser.add_argument('-c', '--conf', help='path to configuration file', default=None)
argparser.add_argument('-w', '--weights', help='weights path', default=None) 
argparser.add_argument('-e', '--engine', help='engine path', default=None) 
argparser.add_argument('-p', '--cpp', help='enable CPP TRT', action="store_true") 
argparser.add_argument('-g', '--gui', help='enable GUI', action="store_true") 

def _main_(args):

    config_path  = args.conf
    input_path = args.input
    weights_path = args.weights
    engine_path = args.engine
    render_mode = args.gui

    trt_engine = TRTengine(isCppInf=args.cpp)

    if engine_path is None:
        with open(config_path) as config_buffer:    
            config = json.load(config_buffer)

        trt_engine.import_from_weights(config, weights_path)
        trt_engine.save_engine()

        return
    else:
        assert trt_engine.load_engine(engine_path), 'Failed to load engine'

    if not input_path:
        return

    image_paths = []
    if os.path.isdir(input_path): 
        for inp_file in os.listdir(input_path):
            image_paths += [os.path.join(input_path, inp_file)]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG', '.ppm'])]

    processing_count = 0
    sum_time = 0

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)

        start_time = time.time()

        boxes = trt_engine.predict_boxes(image)

        sum_time += time.time() - start_time
        processing_count += 1

        if render_mode:
            draw_boxes(image, boxes, trt_engine.get_labels(), 0.5) 
            cv2.imshow('result', np.uint8(image))
            
            if render_mode and cv2.waitKey(0) == 27:
                break  # esc to quit

    fps = processing_count / sum_time
    print('Result: {}'.format(fps))

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)