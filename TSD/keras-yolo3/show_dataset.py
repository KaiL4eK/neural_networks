#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from yolo import create_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard, MAP_evaluation

import tensorflow as tf
import keras
from keras.models import load_model

import keras.backend as K

import cv2

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    train_ints, _ = parse_voc_annotation(config['train']['train_annot_folder'], config['train']['train_image_folder'], config['train']['cache_name'], config['model']['labels'])
    valid_ints, _ = parse_voc_annotation(config['valid']['valid_annot_folder'], config['valid']['valid_image_folder'], config['valid']['cache_name'], config['model']['labels'])

    instances = train_ints + valid_ints
    max_box_per_image = max([len(inst['object']) for inst in instances])

    valid_generator = BatchGenerator(
        instances           = instances, 
        anchors             = config['model']['anchors'],   
        labels              = config['model']['labels'],        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = 1,
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = None,
        isValid             = True
    )

    for i in range(len(instances)):
        img_batch = valid_generator[i]
        img = img_batch[0]

        # img = cv2.resize(img, (900, 900))

        cv2.imshow('1', img.astype(np.uint8))
        cv2.waitKey(0)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   

    args = argparser.parse_args()
    _main_(args)
