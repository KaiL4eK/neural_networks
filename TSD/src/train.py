#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
import yolo
import tensorflow as tf
from generator import BatchGenerator
from utils.utils import normalize, makedirs, init_session, unfreeze_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow.keras.optimizers as opt
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

from _common import utils
import _common.callbacks as cbs
from _common.voc import replace_all_labels_2_one, create_training_instances

# MBN2 - 149 ms/step
def prepare_generators(config):
    if config['train']['cache_name']:
        makedirs(os.path.dirname(config['train']['cache_name']))

    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['annot_folder'],
        config['train']['image_folder'],
        config['train']['cache_name'],
        config['valid']['annot_folder'],
        config['valid']['image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )

    # Dirty hack
    train_ints, train_labels = replace_all_labels_2_one(train_ints, 'sign')
    valid_ints, valid_labels = replace_all_labels_2_one(valid_ints, 'sign')
    labels = list(train_labels.keys())

    print('\nTraining on: \t{}\n'.format(labels))
    print('\nSamples: {} / {}\t\n'.format(len(train_ints), len(valid_ints)))

    ###############################
    #   Create the generators
    ###############################
    train_generator = BatchGenerator(
        instances=train_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=config['model']['downsample'],  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['train']['min_input_size'],
        max_net_size=config['train']['max_input_size'],
        shuffle=True,
        jitter=0.1,
        norm=normalize,
        tile_count=config['model']['tiles']
    )

    valid_generator = BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=config['model']['downsample'],  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        norm=normalize,
        infer_sz=config['model']['infer_shape'],
        tile_count=config['model']['tiles']
    )

    config['train']['mbpi'] = max_box_per_image
    config['model']['labels'] = labels

    return train_generator, valid_generator


def prepare_model(config, initial_weights):

    if initial_weights and os.path.exists(initial_weights):
        freezing = False

    yolo_model = yolo.YOLO_Model(
        config['model'],
        config['train']
    )

    # load the pretrained weight if exists, otherwise load the backend weight only
    if initial_weights:
        yolo_model.load_weights(initial_weights)

    return yolo_model


def train_freezed(config, train_model, train_generator, valid_generator):
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=3,
        mode='min',
        verbose=1
    )

    callbacks = [early_stop]

    if freezing:
        optimizer = Adam(lr=1e-3)
        train_model.compile(loss=yolo.dummy_loss, optimizer=optimizer)
        train_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train_generator) *
            config['train']['train_times'],

            validation_data=valid_generator,
            validation_steps=len(valid_generator) *
            config['valid']['valid_times'],

            epochs=config['train']['nb_epochs'] +
            config['train']['warmup_epochs'],
            verbose=2 if config['train']['debug'] else 1,
            callbacks=callbacks,
            workers=8,
            max_queue_size=100
        )

    # make a GPU version of infer_model for evaluation
    # if multi_gpu > 1:
    #     infer_model = load_model(config['train']['saved_weights_name'])

    unfreeze_model(infer_model)


def start_train(
        config, 
        yolo_model: yolo.YOLO_Model, 
        train_generator, 
        valid_generator
    ):
    print('Full training')

    ###############################
    #   Optimizers
    ###############################

    optimizers = {
        'SGD': opt.SGD(lr=config['train']['learning_rate']),
        'Adam': opt.Adam(lr=config['train']['learning_rate']),
        'Nadam': opt.Nadam(lr=config['train']['learning_rate']),
        'RMSprop': opt.RMSprop(lr=config['train']['learning_rate']),
    }

    optimizer = optimizers[config['train']['optimizer']]

    if config['train']['clipnorm'] > 0:
        optimizer.clipnorm = config['train']['clipnorm']
    
    if config['train'].get('lr_decay', 0) > 0:
        optimizer.decay = config['train']['lr_decay']
    
    if config['train']['optimizer'] == 'Nadam':
        # Just to set field
        optimizer.decay = 0.0

    ###############################
    #   Callbacks
    ###############################
    
    checkpoint_name = utils.get_checkpoint_name(config)
    utils.makedirs_4_file(checkpoint_name)

    checkpoint_vloss = cbs.CustomModelCheckpoint(
        model_to_save=yolo_model.infer_model,
        filepath=checkpoint_name,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )

    tensorboard_logdir = utils.get_tensorboard_name(config)
    utils.makedirs(tensorboard_logdir)
    print('Tensorboard dir: {}'.format(tensorboard_logdir))

    tensorboard_cb = TensorBoard(
        log_dir=tensorboard_logdir,
        histogram_freq=0,
        write_graph=False
    )

    mAP_checkpoint_name = utils.get_mAP_checkpoint_name(config)
    utils.makedirs_4_file(mAP_checkpoint_name)

    map_evaluator_cb = cbs.MAP_evaluation(
        model=yolo_model,
        generator=valid_generator,
        save_best=True,
        save_name=mAP_checkpoint_name,
        tensorboard=tensorboard_cb
    )

    reduce_on_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=20,
        verbose=1,
        mode='min',
        min_delta=0,
        cooldown=10,
        min_lr=1e-5
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=80,
        mode='min',
        verbose=1
    )

    # logger_cb = cbs.CustomLogger(
    #     config=config,
    #     tensorboard=tensorboard_cb
    # )

    # fps_logger = cbs.FPSLogger(
    #     infer_model=yolo_model.infer_model,
    #     generator=valid_generator,
    #     infer_sz=config['model']['infer_shape'],
    #     tensorboard=tensorboard_cb
    # )

    callbacks = [tensorboard_cb, map_evaluator_cb, early_stop]
    callbacks += [reduce_on_plateau]
    # callbacks += [fps_logger]
    # callbacks += [checkpoint_vloss]

    ###############################
    #   Prepare fit
    ###############################

    yolo_model.train_model.compile(loss=yolo.dummy_loss, optimizer=optimizer)
    yolo_model.train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],

        validation_data=valid_generator,
        validation_steps=len(valid_generator) * config['valid']['valid_times'],

        epochs=config['train']['nb_epochs'],
        verbose=1,
        callbacks=callbacks,
        workers=8,
        max_queue_size=100,
        use_multiprocessing=False
    )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-w', '--weights', help='path to pretrained model', default=None)
    args = argparser.parse_args()

    init_session(1.0)

    config_path = args.conf
    initial_weights = args.weights

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    train_generator, valid_generator = prepare_generators(config)

    yolo_model = prepare_model(config, initial_weights)

    # if freezing:
        # train_freezed(config, train_model, train_generator, valid_generator)

    start_train(config, yolo_model, train_generator, valid_generator)

    clear_session()
