#! /usr/bin/env python

# Append library path
import sys
sys.path.append('ext_repos/keras-yolo2')

from preprocessing import parse_annotation, parse_annotation_csv
from frontend import YOLO
from datetime import datetime
import numpy as np
import tensorflow as tf
import shutil
import json
import keras
import argparse
import os

import common as cmn
import keras.backend as K

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
    '-d',
    '--debug',
    action='store_true',
    default=False,
    help='debug')

def init_session():
    config = tf.ConfigProto()

    config.gpu_options.allow_growth                     = True
    # config.gpu_options.per_process_gpu_memory_fraction  = 0.9

    # cmn.setCHWDataFormat()

    if cmn.isDataFormatCv():
        print('Data format: HWC')
    else:
        print('Data format: CWH')

    K.set_session(tf.Session(config=config))


def _main_(args):
    config_path = args.conf
    debug_enabled = args.debug
    
    init_session()

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'], 
                                                config['train']['train_image_folder'], 
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
                                                    config['valid']['valid_image_folder'], 
                                                    config['model']['labels'])
        split = False
    else:
        split = True

    if split:
        print('No validation, split training images: %d / %d' % (0.8 * len(train_imgs), len(train_imgs)) )

        train_valid_split = int(0.8*len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    print('Training / Validation images: %d / %d' % (len(train_imgs), len(valid_imgs)) )

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)           

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        print('Seen labels:\t', train_labels)
        config['model']['labels'] = train_labels.keys()
        with open("labels.json", 'w') as outfile:
            json.dump({"labels" : list(train_labels.keys())},outfile)

    
    max_box_per_image = config['model']['max_box_per_image']
    max_box_per_image_calc = max([len(inst['object']) for inst in (train_imgs + valid_imgs)])

    if max_box_per_image < max_box_per_image_calc:
        print('Invalid max_box_per_image value')
        exit(1)

    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = (config['model']['input_size_h'], config['model']['input_size_w']), 
                labels              = config['model']['labels'], 
                anchors             = config['model']['anchors'],
                trainable           = config['model']['trainable'],
                gray_mode           = config['model']['gray_mode'],
                max_box_per_image   = max_box_per_image )

    ###############################
    #   Load the pretrained weights (if any) 
    ###############################    

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        try:
            yolo.load_weights(config['train']['pretrained_weights'])
        except Exception as e:
            print(e)
            print("Failed to load pretrained weights")
            print("Continue from scratch!")

    ###############################
    #   Start the training process 
    ###############################

    tensorboard_log_dir_prefix = config['train']['tensorboard_log_dir']
    tensorboard_log_dir_idx = 0

    while True:

        tensorboard_log_dir = tensorboard_log_dir_prefix + '-{}'.format(tensorboard_log_dir_idx)

        if not os.path.exists( tensorboard_log_dir ): 
            print('Set Tensorboard log dir:', tensorboard_log_dir)
            break

        tensorboard_log_dir_idx += 1

    yolo.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epochs          = config['train']['nb_epochs'], 
               learning_rate      = config['train']['learning_rate'], 
               batch_size         = config['train']['batch_size'],
               warmup_epochs      = config['train']['warmup_epochs'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               saved_weights_name = config['train']['saved_weights_name'],
               debug              = debug_enabled,
               early_stop         = config['train']['early_stop'],
               workers            = config['train']['workers'],
               max_queue_size     = config['train']['max_queue_size'],
               tb_logdir          = tensorboard_log_dir)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
