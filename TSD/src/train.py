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
import tensorflow.keras.optimizers as opt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

from _common import utils
import _common.callbacks as cbs
from _common.voc import parse_voc_annotation, split_by_objects, replace_all_labels_2_one, create_training_instances

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
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.1,
        norm=normalize
    )

    valid_generator = BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=config['model']['downsample'],  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        norm=normalize,
        infer_sz=config['model']['infer_shape']
    )

    config['other'] = {
        'labels': labels,
        'mbpi': max_box_per_image
    }

    return train_generator, valid_generator


def prepare_model(config, initial_weights):

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))

    freezing = config['train'].get('freeze', True)
    config['train']['warmup_epochs'] = 0

    # warmup_batches = config['train']['warmup_epochs'] * \
    #     (config['train']['train_times'] * len(train_generator))

    if initial_weights and os.path.exists(initial_weights):
        freezing = False

    train_model, infer_model, _ = yolo.create_model_new(
        nb_class=len(config['other']['labels']),
        anchors=config['model']['anchors'],
        max_box_per_image=config['other']['mbpi'],
        max_input_size=config['model']['max_input_size'],
        batch_size=config['train']['batch_size'],
        warmup_batches=0,
        ignore_thresh=config['train']['ignore_thresh'],
        multi_gpu=multi_gpu,
        grid_scales=config['train']['grid_scales'],
        obj_scale=config['train']['obj_scale'],
        noobj_scale=config['train']['noobj_scale'],
        xywh_scale=config['train']['xywh_scale'],
        class_scale=config['train']['class_scale'],
        base=config['model']['base'],
        anchors_per_output=config['model']['anchors_per_output'],
        is_freezed=freezing,
        load_src_weights=config['train'].get('load_src_weights', True)
    )

    model_render_file = 'images/{}.png'.format(config['model']['base'])
    if not os.path.isdir(os.path.dirname(model_render_file)):
        os.makedirs(os.path.dirname(model_render_file))
    # plot_model(infer_model, to_file=model_render_file, show_shapes=True)
    # infer_model.summary()

    # load the pretrained weight if exists, otherwise load the backend weight only
    if initial_weights and os.path.exists(initial_weights):
        print("\nLoading pretrained weights {}".format(initial_weights))
        train_model.load_weights(
            initial_weights, by_name=True, skip_mismatch=True)

    return train_model, infer_model, freezing


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


def start_train(config, train_model, infer_model, train_generator, valid_generator):
    print('Full training')

    checkpoint_name = utils.get_checkpoint_name(config)
    utils.makedirs_4_file(checkpoint_name)

    checkpoint_vloss = cbs.CustomModelCheckpoint(
        model_to_save=infer_model,
        filepath=checkpoint_name,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )

    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=10,
        verbose=1,
        mode='min',
        min_delta=0.01,
        cooldown=0,
        min_lr=0
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
        infer_model=infer_model,
        generator=valid_generator,
        save_best=True,
        save_name=mAP_checkpoint_name,
        tensorboard=tensorboard_cb,
        iou_threshold=0.5,
        score_threshold=0.5,
        infer_sz=config['model']['infer_shape'],
        evaluate=evaluate
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        mode='min',
        verbose=1
    )

    logger_cb = cbs.CustomLogger(
        config=config,
        tensorboard=tensorboard_cb
    )

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

    callbacks = [checkpoint_vloss, tensorboard_cb, map_evaluator_cb, logger_cb]

    ###############################
    #   Prepare fit
    ###############################

    train_model.compile(loss=yolo.dummy_loss, optimizer=optimizer)
    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],

        validation_data=valid_generator,
        validation_steps=len(valid_generator) * config['valid']['valid_times'],

        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=2 if config['train']['debug'] else 1,
        callbacks=callbacks,
        workers=8,
        max_queue_size=100
    )

    ###############################
    #   Run the evaluation
    ###############################
    # compute mAP for all the classes
    average_precisions = evaluate(model=infer_model,
                                  generator=valid_generator,
                                  iou_threshold=0.5,
                                  net_h=config['model']['infer_shape'][0],
                                  net_w=config['model']['infer_shape'][1])

    # print the score
    for label, average_precision in average_precisions.items():
        print(label + ': {:.4f}'.format(average_precision))
    print('Last mAP: {:.4f}'.format(
        sum(average_precisions.values()) / len(average_precisions)))


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

    train_model, infer_model, freezing = prepare_model(config, initial_weights)

    if freezing:
        train_freezed(config, train_model, train_generator, valid_generator)

    start_train(config, train_model, infer_model, train_generator, valid_generator)

    clear_session()