#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
from utils import list_images
import json
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input


    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    if weights_path == '':
        weights_path = config['train']['saved_weights_name']

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = (config['model']['input_size_h'],config['model']['input_size_w']), 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'],
                gray_mode           = config['model']['gray_mode'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if not image_path[-4:] in ['.jpg', '.png', '.jpeg']:
        print('File type {} is not supported'.format(image_path[-4:]))
    elif not os.path.isfile(image_path):
        print('Input is not file')    
            
    else:  
        image = cv2.imread(image_path)
        n_passes = 1000

        full_time = 0
        start = time.time()
        
        for i in range(n_passes):
            _, pred_time = yolo.predict_time(image)
            full_time += pred_time

        elapsed = time.time() - start

        print("10 full predictions: {} [s], 1 full prediction: {} [s], 1 prediction only: {} [s] \
                    FPS: {}".format( elapsed, elapsed * 1. / n_passes, full_time * 1. / n_passes, n_passes * 1. / elapsed ))

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
