#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs

import cv2


def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    makedirs(os.path.dirname(config['train']['cache_name']))

    train_ints, labels = parse_voc_annotation(config['train']['train_annot_folder'],
                                         config['train']['train_image_folder'],
                                         config['train']['cache_name'],
                                         config['model']['labels'])
    # valid_ints, _ = parse_voc_annotation(config['valid']['valid_annot_folder'], config['valid']['valid_image_folder'], config['valid']['cache_name'], config['model']['labels'])

    instances = train_ints
    max_box_per_image = max([len(inst['object']) for inst in instances])

    print(list(labels.keys()))

    valid_generator = BatchGenerator(
        instances           = instances,
        anchors             = config['model']['anchors'],   
        labels              = list(labels.keys()),
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = 1,
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = None
    )

    for i in range(len(instances)):
        img_batch = valid_generator[i]
        img = img_batch[0]

        img = cv2.resize(img, (640, 480))

        cv2.imshow('1', img.astype(np.uint8))
        if cv2.waitKey(0) == 27:
            break


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   

    args = argparser.parse_args()
    _main_(args)
