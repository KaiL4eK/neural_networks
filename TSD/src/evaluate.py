#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
import yolo
from generator import BatchGenerator
from utils.utils import normalize, evaluate, init_session
from tensorflow.keras.models import load_model

from _common.voc import parse_voc_annotation, replace_all_labels_2_one
import _common.utils as c_ut


def _main_(args):
    config_path = args.conf
    weights = args.weights

    init_session(1.0)

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Create the validation generator
    ###############################  
    valid_ints, labels = parse_voc_annotation(
        config['eval']['annot_folder'], 
        config['eval']['image_folder'], 
        config['eval']['cache_name'],
        config['model']['labels']
    )

    valid_ints, labels = replace_all_labels_2_one(valid_ints, 'sign')
    labels = list(labels.keys())
    
    # labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    # labels = sorted(labels)
   
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = config['model']['downsample'], # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = False, 
        jitter              = 0.0, 
        norm                = normalize
    )

    _, infer_model, _ = yolo.create_model_new(
        nb_class=len(labels),
        anchors=config['model']['anchors'],
        max_box_per_image=0,
        max_input_size=config['model']['max_input_size'],
        batch_size=config['train']['batch_size'],
        warmup_batches=0,
        ignore_thresh=config['train']['ignore_thresh'],
        multi_gpu='0',
        grid_scales=config['train']['grid_scales'],
        obj_scale=config['train']['obj_scale'],
        noobj_scale=config['train']['noobj_scale'],
        xywh_scale=config['train']['xywh_scale'],
        class_scale=config['train']['class_scale'],
        base=config['model']['base'],
        anchors_per_output=config['model']['anchors_per_output'],
        is_freezed=False,
        load_src_weights=config['train'].get('load_src_weights', True)
    )

    infer_model.load_weights(weights)

    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, 
                                  valid_generator,
                                  iou_threshold=0.5,
                                  obj_thresh=0.5,
                                  nms_thresh=0.45,
                                  net_h=config['model']['infer_shape'][0],
                                  net_w=config['model']['infer_shape'][1],
                                  verbose=True,
                                  save_path=None)
    c_ut.print_predicted_average_precisions(average_precisions)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate YOLOv3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')    
    argparser.add_argument('-w', '--weights', help='path to pretrained model')

    args = argparser.parse_args()
    _main_(args)
