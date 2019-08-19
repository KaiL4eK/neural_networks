#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
import yolo
import tensorflow as tf
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs, init_session, unfreeze_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

from _common import utils
from _common.callbacks import CustomModelCheckpoint, CustomTensorBoard, MAP_evaluation
from _common.voc import parse_voc_annotation, split_by_objects, replace_all_labels_2_one, create_training_instances

# MBN2 - 149 ms/step
from train import prepare_generators, prepare_model, train_freezed, start_train


def train_grid_search(config, initial_weights, lr_list, optimizer_list):
    train_generator, valid_generator = prepare_generators(config)

    for lr in lr_list:
        for optimizer in optimizer_list:
            config['train']['learning_rate'] = lr
            config['train']['optimizer'] = optimizer

            train_model, infer_model, freezing = prepare_model(
                config, initial_weights)

            if freezing:
                train_freezed(config, train_model,
                              train_generator, valid_generator)

            print('Training with {} / {}'.format(lr, optimizer))
            start_train(config, train_model, infer_model,
                  train_generator, valid_generator)

            clear_session()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument(
        '-w', '--weights', help='path to pretrained model', default=None)
    args = argparser.parse_args()

    init_session(1.0)

    config_path = args.conf
    initial_weights = args.weights

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    train_grid_search(config, initial_weights, [
                      3e-4, 7e-4], ['Adam'])
